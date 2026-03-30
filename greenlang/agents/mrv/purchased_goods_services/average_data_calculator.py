# -*- coding: utf-8 -*-
"""
Average-Data Calculator Engine -- AGENT-MRV-014 Engine 3 of 7

This module implements the AverageDataCalculatorEngine for the Purchased
Goods & Services Agent (GL-MRV-S3-001).  The engine calculates GHG
emissions using the *average-data method* from the GHG Protocol Scope 3
Technical Guidance, Section 1.3, where emissions are determined by
multiplying physical quantities of purchased goods by cradle-to-gate
emission factors from LCA databases.

Core formula::

    Emissions_kgCO2e = Quantity_kg x EF_kgCO2e_per_kg x WasteLossFactor

When the emission factor excludes transport to the reporting company's
gate (``includes_transport=False``), a transport adder is calculated::

    Transport_kgCO2e = (Quantity_kg / 1000) x Distance_km x EF_transport

The engine supports:

- **Single-item calculation** via ``calculate_single``.
- **Record-based calculation** via ``calculate_from_record`` (pre-enriched
  ``PhysicalRecord`` inputs).
- **Batch processing** via ``calculate_batch`` and
  ``calculate_batch_records``.
- **Unit conversion** across 20+ units (mass, volume, piece) to kg using
  deterministic ``Decimal`` arithmetic.
- **Material key resolution** from ``ProcurementItem`` attributes
  (``material_category``, ``description``, ``metadata``).
- **Multi-material allocation** for BOM-based products with per-material
  emission factor application and weighted aggregation.
- **Transport emission adder** for 7 transport modes (road, rail, sea,
  air, barge, pipeline, truck).
- **5-dimension DQI scoring** per GHG Protocol Scope 3 Standard
  Chapter 7 with composite score and quality tier.
- **Aggregation by material** for portfolio-level reporting.
- **Coverage analysis** with success/failure/skip counts.
- **Uncertainty estimation** via pedigree matrix and analytical
  propagation from UNCERTAINTY_RANGES for average-data method.
- **SHA-256 provenance tracking** for complete audit trails.
- **Prometheus metrics** for calculation counts, durations, and emissions.
- **Health check** for operational monitoring.

Thread Safety:
    The engine is implemented as a thread-safe singleton using
    ``threading.RLock`` with double-checked locking.  All internal
    state is either immutable (constant tables) or protected by the
    instance lock where necessary.

Zero-Hallucination Guarantee:
    All numeric operations use ``Decimal`` with explicit quantization.
    No LLM calls are made for any numeric calculation.  Emission
    factors are sourced exclusively from the deterministic
    ``PHYSICAL_EMISSION_FACTORS`` constant table in ``models.py``.

Emission Factor Sources:
    - World Steel Association 2023
    - International Aluminium Institute 2023
    - PlasticsEurope Eco-profiles 2022
    - ICE Database v3.0 (University of Bath)
    - CEPI (Confederation of European Paper Industries) 2022
    - Textile Exchange 2023
    - ICA (International Copper Association) 2022

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.mrv.purchased_goods_services.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    ZERO,
    ONE,
    ONE_HUNDRED,
    ONE_THOUSAND,
    DECIMAL_PLACES,
    CalculationMethod,
    PhysicalEFSource,
    MaterialCategory,
    DQIDimension,
    DQIScore,
    EmissionGas,
    ProcurementType,
    PHYSICAL_EMISSION_FACTORS,
    DQI_SCORE_VALUES,
    UNCERTAINTY_RANGES,
    PEDIGREE_UNCERTAINTY_FACTORS,
    ProcurementItem,
    PhysicalRecord,
    AverageDataResult,
    PhysicalEF,
    DQIAssessment,
)
from greenlang.agents.mrv.purchased_goods_services.config import (
    PurchasedGoodsServicesConfig,
)
from greenlang.agents.mrv.purchased_goods_services.metrics import (
    PurchasedGoodsServicesMetrics,
)
from greenlang.agents.mrv.purchased_goods_services.provenance import (
    PurchasedGoodsProvenanceTracker,
    ProvenanceStage,
    hash_physical_record,
    hash_physical_ef,
    hash_average_data_result,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AverageDataCalculatorEngine",
    "UNIT_CONVERSION_TO_KG",
    "TRANSPORT_EFS",
    "CATEGORY_DEFAULT_MATERIAL",
    "MATERIAL_DENSITY_KG_PER_L",
    "MATERIAL_CATEGORY_MAP",
    "DESCRIPTION_KEYWORD_MAP",
    "MATERIAL_UNCERTAINTY_PCT",
]

# ============================================================================
# Embedded Data Tables -- All Decimal for Deterministic Arithmetic
# ============================================================================

# ---------------------------------------------------------------------------
# Unit conversion factors to kilograms
# Covers mass units, volume units (require density lookup), and count units.
# Volume units map to a generic density of 1.0 kg/L unless overridden by
# material-specific density via MATERIAL_DENSITY_KG_PER_L.
# ---------------------------------------------------------------------------

UNIT_CONVERSION_TO_KG: Dict[str, Decimal] = {
    # -- Mass units --
    "kg": Decimal("1"),
    "kilogram": Decimal("1"),
    "kilograms": Decimal("1"),
    "g": Decimal("0.001"),
    "gram": Decimal("0.001"),
    "grams": Decimal("0.001"),
    "mg": Decimal("0.000001"),
    "milligram": Decimal("0.000001"),
    "milligrams": Decimal("0.000001"),
    "tonne": Decimal("1000"),
    "tonnes": Decimal("1000"),
    "t": Decimal("1000"),
    "metric_ton": Decimal("1000"),
    "metric_tons": Decimal("1000"),
    "lb": Decimal("0.45359237"),
    "lbs": Decimal("0.45359237"),
    "pound": Decimal("0.45359237"),
    "pounds": Decimal("0.45359237"),
    "oz": Decimal("0.02834952"),
    "ounce": Decimal("0.02834952"),
    "ounces": Decimal("0.02834952"),
    "ton_us": Decimal("907.18474"),
    "short_ton": Decimal("907.18474"),
    "ton_uk": Decimal("1016.0469"),
    "long_ton": Decimal("1016.0469"),
    # -- Volume units (default density = 1.0 kg/L for water-like) --
    "liter": Decimal("1"),
    "litre": Decimal("1"),
    "l": Decimal("1"),
    "ml": Decimal("0.001"),
    "milliliter": Decimal("0.001"),
    "millilitre": Decimal("0.001"),
    "m3": Decimal("1000"),
    "cubic_meter": Decimal("1000"),
    "cubic_metre": Decimal("1000"),
    "gallon_us": Decimal("3.78541"),
    "gal_us": Decimal("3.78541"),
    "gallon_uk": Decimal("4.54609"),
    "gal_uk": Decimal("4.54609"),
    "barrel": Decimal("158.987"),
    "bbl": Decimal("158.987"),
    # -- Count / piece units (require per-piece weight in metadata) --
    "piece": Decimal("1"),
    "pieces": Decimal("1"),
    "pcs": Decimal("1"),
    "unit": Decimal("1"),
    "units": Decimal("1"),
    "each": Decimal("1"),
    "ea": Decimal("1"),
    "pair": Decimal("2"),
    "pairs": Decimal("2"),
    "dozen": Decimal("12"),
    "gross": Decimal("144"),
}

#: Units that represent volume (require density-based conversion).
_VOLUME_UNITS: frozenset = frozenset({
    "liter", "litre", "l", "ml", "milliliter", "millilitre",
    "m3", "cubic_meter", "cubic_metre",
    "gallon_us", "gal_us", "gallon_uk", "gal_uk",
    "barrel", "bbl",
})

#: Units that represent count/pieces (require per-piece weight).
_COUNT_UNITS: frozenset = frozenset({
    "piece", "pieces", "pcs", "unit", "units", "each", "ea",
    "pair", "pairs", "dozen", "gross",
})

# ---------------------------------------------------------------------------
# Material densities in kg per liter (for volume-to-mass conversion)
# Used when the input unit is a volume unit and material_key is known.
# ---------------------------------------------------------------------------

MATERIAL_DENSITY_KG_PER_L: Dict[str, Decimal] = {
    # Liquids and semi-liquids
    "ammonia": Decimal("0.73"),
    "ethylene": Decimal("0.57"),
    "propylene": Decimal("0.61"),
    "methanol": Decimal("0.79"),
    "natural_rubber": Decimal("0.92"),
    "synthetic_rubber_sbr": Decimal("0.94"),
    # Fuels (for reference, not typically Cat 1)
    "diesel": Decimal("0.84"),
    "gasoline": Decimal("0.74"),
    "kerosene": Decimal("0.80"),
    # Water-based default
    "water": Decimal("1.00"),
    # Plastics (molten/liquid state)
    "hdpe": Decimal("0.95"),
    "ldpe": Decimal("0.92"),
    "pp_polypropylene": Decimal("0.90"),
    "pet": Decimal("1.38"),
    "pvc": Decimal("1.40"),
    "ps_polystyrene": Decimal("1.05"),
    "abs": Decimal("1.05"),
    "nylon_6": Decimal("1.14"),
}

# ---------------------------------------------------------------------------
# Transport emission factors in kgCO2e per tonne-km
# Sources: DEFRA 2023 conversion factors, EcoTransIT World 2023
# ---------------------------------------------------------------------------

TRANSPORT_EFS: Dict[str, Decimal] = {
    "road": Decimal("0.0620"),
    "truck": Decimal("0.0620"),
    "rail": Decimal("0.0220"),
    "train": Decimal("0.0220"),
    "sea": Decimal("0.0080"),
    "ship": Decimal("0.0080"),
    "ocean": Decimal("0.0080"),
    "air": Decimal("0.6020"),
    "flight": Decimal("0.6020"),
    "barge": Decimal("0.0310"),
    "inland_waterway": Decimal("0.0310"),
    "pipeline": Decimal("0.0050"),
}

# ---------------------------------------------------------------------------
# Default material key for each MaterialCategory
# Used when no specific material_key is available on the ProcurementItem.
# ---------------------------------------------------------------------------

CATEGORY_DEFAULT_MATERIAL: Dict[MaterialCategory, str] = {
    MaterialCategory.RAW_METALS: "steel_world_avg",
    MaterialCategory.PLASTICS: "hdpe",
    MaterialCategory.CHEMICALS: "ammonia",
    MaterialCategory.PAPER: "kraft_paper",
    MaterialCategory.TEXTILES: "cotton_conventional",
    MaterialCategory.ELECTRONICS: "pcb_printed_circuit",
    MaterialCategory.FOOD: "corrugated_cardboard",
    MaterialCategory.PACKAGING: "corrugated_cardboard",
    MaterialCategory.CONSTRUCTION: "concrete_readymix_30mpa",
    MaterialCategory.MACHINERY: "steel_world_avg",
    MaterialCategory.FUELS: "methanol",
    MaterialCategory.MINERALS: "cement_portland_global",
    MaterialCategory.GLASS: "glass_general",
    MaterialCategory.RUBBER: "natural_rubber",
    MaterialCategory.WOOD: "timber_softwood_sawn",
    MaterialCategory.AGRICULTURE: "ammonia",
    MaterialCategory.SERVICES_IT: "pcb_printed_circuit",
    MaterialCategory.SERVICES_PROFESSIONAL: "kraft_paper",
    MaterialCategory.SERVICES_FINANCIAL: "kraft_paper",
    MaterialCategory.OTHER: "steel_world_avg",
}

# ---------------------------------------------------------------------------
# Material category to material key mapping (reverse index)
# Maps PHYSICAL_EMISSION_FACTORS keys to their MaterialCategory.
# ---------------------------------------------------------------------------

MATERIAL_CATEGORY_MAP: Dict[str, MaterialCategory] = {
    # Metals
    "steel_primary_bof": MaterialCategory.RAW_METALS,
    "steel_secondary_eaf": MaterialCategory.RAW_METALS,
    "steel_world_avg": MaterialCategory.RAW_METALS,
    "steel_virgin_100pct": MaterialCategory.RAW_METALS,
    "aluminum_primary_global": MaterialCategory.RAW_METALS,
    "aluminum_secondary": MaterialCategory.RAW_METALS,
    "aluminum_33pct_recycled": MaterialCategory.RAW_METALS,
    "copper_primary": MaterialCategory.RAW_METALS,
    "lead": MaterialCategory.RAW_METALS,
    "zinc": MaterialCategory.RAW_METALS,
    "lithium_carbonate": MaterialCategory.RAW_METALS,
    # Plastics
    "hdpe": MaterialCategory.PLASTICS,
    "ldpe": MaterialCategory.PLASTICS,
    "pp_polypropylene": MaterialCategory.PLASTICS,
    "pet": MaterialCategory.PLASTICS,
    "pvc": MaterialCategory.PLASTICS,
    "ps_polystyrene": MaterialCategory.PLASTICS,
    "abs": MaterialCategory.PLASTICS,
    "nylon_6": MaterialCategory.PLASTICS,
    # Construction
    "cement_portland_global": MaterialCategory.CONSTRUCTION,
    "cement_portland_cem_i": MaterialCategory.CONSTRUCTION,
    "concrete_readymix_30mpa": MaterialCategory.CONSTRUCTION,
    "concrete_high_50mpa": MaterialCategory.CONSTRUCTION,
    "float_glass": MaterialCategory.GLASS,
    "glass_general": MaterialCategory.GLASS,
    "bricks_general": MaterialCategory.CONSTRUCTION,
    "timber_softwood_sawn": MaterialCategory.WOOD,
    "timber_hardwood_sawn": MaterialCategory.WOOD,
    "timber_glulam": MaterialCategory.WOOD,
    # Paper and Packaging
    "corrugated_cardboard": MaterialCategory.PAPER,
    "kraft_paper": MaterialCategory.PAPER,
    "recycled_paper": MaterialCategory.PAPER,
    # Textiles
    "cotton_conventional": MaterialCategory.TEXTILES,
    "cotton_organic": MaterialCategory.TEXTILES,
    "polyester_fiber": MaterialCategory.TEXTILES,
    "nylon_fiber": MaterialCategory.TEXTILES,
    "wool": MaterialCategory.TEXTILES,
    # Electronics
    "silicon_wafer_solar": MaterialCategory.ELECTRONICS,
    "pcb_printed_circuit": MaterialCategory.ELECTRONICS,
    # Chemicals
    "ammonia": MaterialCategory.CHEMICALS,
    "ethylene": MaterialCategory.CHEMICALS,
    "propylene": MaterialCategory.CHEMICALS,
    "methanol": MaterialCategory.CHEMICALS,
    # Rubber
    "natural_rubber": MaterialCategory.RUBBER,
    "synthetic_rubber_sbr": MaterialCategory.RUBBER,
}

# ---------------------------------------------------------------------------
# Description keyword to material key mapping
# Used for fuzzy material resolution from item descriptions when no
# explicit material_key or material_category is available.
# ---------------------------------------------------------------------------

DESCRIPTION_KEYWORD_MAP: Dict[str, str] = {
    "steel": "steel_world_avg",
    "stainless steel": "steel_primary_bof",
    "aluminium": "aluminum_primary_global",
    "aluminum": "aluminum_primary_global",
    "copper": "copper_primary",
    "lead": "lead",
    "zinc": "zinc",
    "lithium": "lithium_carbonate",
    "hdpe": "hdpe",
    "ldpe": "ldpe",
    "polypropylene": "pp_polypropylene",
    "pp ": "pp_polypropylene",
    "pet ": "pet",
    "polyethylene terephthalate": "pet",
    "pvc": "pvc",
    "polystyrene": "ps_polystyrene",
    "abs ": "abs",
    "nylon": "nylon_6",
    "cement": "cement_portland_global",
    "concrete": "concrete_readymix_30mpa",
    "glass": "glass_general",
    "brick": "bricks_general",
    "timber": "timber_softwood_sawn",
    "lumber": "timber_softwood_sawn",
    "plywood": "timber_softwood_sawn",
    "softwood": "timber_softwood_sawn",
    "hardwood": "timber_hardwood_sawn",
    "glulam": "timber_glulam",
    "cardboard": "corrugated_cardboard",
    "corrugated": "corrugated_cardboard",
    "kraft": "kraft_paper",
    "paper": "kraft_paper",
    "recycled paper": "recycled_paper",
    "cotton": "cotton_conventional",
    "organic cotton": "cotton_organic",
    "polyester": "polyester_fiber",
    "wool": "wool",
    "ammonia": "ammonia",
    "ethylene": "ethylene",
    "propylene": "propylene",
    "methanol": "methanol",
    "rubber": "natural_rubber",
    "natural rubber": "natural_rubber",
    "synthetic rubber": "synthetic_rubber_sbr",
    "silicon": "silicon_wafer_solar",
    "pcb": "pcb_printed_circuit",
    "circuit board": "pcb_printed_circuit",
}

# ---------------------------------------------------------------------------
# Material-specific uncertainty percentages (+/-)
# Source: ecoinvent pedigree matrix, ICE v3.0 uncertainty estimates
# ---------------------------------------------------------------------------

MATERIAL_UNCERTAINTY_PCT: Dict[str, Decimal] = {
    # Metals -- well-characterised processes
    "steel_primary_bof": Decimal("15"),
    "steel_secondary_eaf": Decimal("20"),
    "steel_world_avg": Decimal("25"),
    "steel_virgin_100pct": Decimal("15"),
    "aluminum_primary_global": Decimal("20"),
    "aluminum_secondary": Decimal("25"),
    "aluminum_33pct_recycled": Decimal("22"),
    "copper_primary": Decimal("20"),
    "lead": Decimal("25"),
    "zinc": Decimal("22"),
    "lithium_carbonate": Decimal("30"),
    # Plastics
    "hdpe": Decimal("20"),
    "ldpe": Decimal("20"),
    "pp_polypropylene": Decimal("20"),
    "pet": Decimal("20"),
    "pvc": Decimal("22"),
    "ps_polystyrene": Decimal("22"),
    "abs": Decimal("25"),
    "nylon_6": Decimal("25"),
    # Construction
    "cement_portland_global": Decimal("15"),
    "cement_portland_cem_i": Decimal("15"),
    "concrete_readymix_30mpa": Decimal("20"),
    "concrete_high_50mpa": Decimal("20"),
    "float_glass": Decimal("25"),
    "glass_general": Decimal("25"),
    "bricks_general": Decimal("25"),
    "timber_softwood_sawn": Decimal("30"),
    "timber_hardwood_sawn": Decimal("30"),
    "timber_glulam": Decimal("30"),
    # Paper
    "corrugated_cardboard": Decimal("20"),
    "kraft_paper": Decimal("20"),
    "recycled_paper": Decimal("25"),
    # Textiles
    "cotton_conventional": Decimal("30"),
    "cotton_organic": Decimal("35"),
    "polyester_fiber": Decimal("25"),
    "nylon_fiber": Decimal("25"),
    "wool": Decimal("35"),
    # Electronics
    "silicon_wafer_solar": Decimal("35"),
    "pcb_printed_circuit": Decimal("40"),
    # Chemicals
    "ammonia": Decimal("15"),
    "ethylene": Decimal("15"),
    "propylene": Decimal("15"),
    "methanol": Decimal("15"),
    # Rubber
    "natural_rubber": Decimal("30"),
    "synthetic_rubber_sbr": Decimal("25"),
}

# ---------------------------------------------------------------------------
# Default per-piece weight in kg (for count/piece unit conversion)
# Used when unit is a count unit and no per_piece_weight_kg in metadata.
# ---------------------------------------------------------------------------

_DEFAULT_PIECE_WEIGHT_KG: Dict[str, Decimal] = {
    "pcb_printed_circuit": Decimal("0.15"),
    "silicon_wafer_solar": Decimal("0.18"),
    "glass_general": Decimal("2.50"),
    "bricks_general": Decimal("3.00"),
    "steel_world_avg": Decimal("10.00"),
}

# ---------------------------------------------------------------------------
# Internal quantize helper
# ---------------------------------------------------------------------------

def _q(value: Decimal, places: int = DECIMAL_PLACES) -> Decimal:
    """Quantize a Decimal value to the configured number of places.

    Uses ROUND_HALF_UP rounding mode for consistency with GHG Protocol
    reporting conventions.

    Args:
        value: The Decimal value to quantize.
        places: Number of decimal places (default from DECIMAL_PLACES).

    Returns:
        Quantized Decimal value.
    """
    if places <= 0:
        return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    quantizer = Decimal(10) ** -places
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)

def _sha256(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of a JSON-serialisable dict.

    Args:
        data: Dictionary to hash.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

# ============================================================================
# AverageDataCalculatorEngine -- Thread-Safe Singleton
# ============================================================================

class AverageDataCalculatorEngine:
    """Engine 3: Average-data emission calculator for Purchased Goods & Services.

    Implements the average-data calculation method from the GHG Protocol
    Scope 3 Technical Guidance Section 1.3.  Emissions are computed by
    multiplying physical quantities (normalised to kg) by industry-average
    cradle-to-gate emission factors sourced from recognised LCA databases.

    This engine handles:

    1. **Unit Conversion** -- Converts 20+ input units (mass, volume,
       count) to kilograms using deterministic Decimal arithmetic.
       Volume units require material-specific density; count units
       require per-piece weight from metadata or defaults.

    2. **Material Key Resolution** -- Resolves the correct material
       emission factor key from item ``material_category``, description
       keywords, or metadata ``material_key`` field.

    3. **Emission Factor Lookup** -- Retrieves cradle-to-gate EFs from
       the ``PHYSICAL_EMISSION_FACTORS`` table (30+ materials, sources
       include World Steel, IAI, PlasticsEurope, ICE v3.0, CEPI).

    4. **Core Calculation** -- ``Qty_kg * EF_kgCO2e/kg * WasteLoss``.

    5. **Transport Adder** -- When ``includes_transport=False``, adds
       ``(Qty_kg/1000) * Distance_km * TransportEF_kgCO2e/tkm``.

    6. **Multi-Material BOM** -- Allocates emissions across BOM
       components with material-specific EFs and mass fractions.

    7. **DQI Scoring** -- 5-dimension scoring (temporal, geographical,
       technological, completeness, reliability) per GHG Protocol
       Chapter 7.

    8. **Batch Processing** -- Processes lists of items/records with
       per-item error isolation and aggregated results.

    9. **Aggregation** -- Groups results by material key for portfolio
       analysis and hot-spot identification.

    10. **Coverage** -- Reports success/failure/skip counts for batch
        operations.

    11. **Uncertainty** -- Estimates uncertainty ranges using the
        pedigree matrix and material-specific uncertainty percentages.

    12. **Provenance** -- SHA-256 hashing of inputs and outputs for
        audit trail integrity.

    Thread Safety:
        Singleton via ``__new__`` with ``threading.RLock``.

    Zero-Hallucination:
        All arithmetic uses ``Decimal``.  No LLM involvement in any
        numeric path.

    Attributes:
        _config: Singleton configuration instance.
        _metrics: Singleton metrics collector.
        _provenance: Singleton provenance tracker.
        _decimal_places: Number of decimal places for quantization.
        _enable_transport: Whether transport adder is enabled.
        _enable_waste: Whether waste/loss factor is enabled.
        _enable_dqi: Whether DQI scoring is enabled.
        _enable_provenance: Whether provenance tracking is enabled.

    Example:
        >>> engine = AverageDataCalculatorEngine()
        >>> item = ProcurementItem(
        ...     description="Hot rolled steel coil",
        ...     spend_amount=Decimal("50000"),
        ...     quantity=Decimal("20000"),
        ...     quantity_unit="kg",
        ...     material_category=MaterialCategory.RAW_METALS,
        ... )
        >>> result = engine.calculate_single(
        ...     item,
        ...     material_key="steel_world_avg",
        ... )
        >>> result.emissions_kgco2e
        Decimal('27400.00000000')
    """

    _instance: Optional[AverageDataCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------

    def __new__(cls) -> AverageDataCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with ``threading.RLock`` to ensure
        thread-safe initialisation.

        Returns:
            The singleton AverageDataCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialise engine from configuration.

        Guarded by ``_initialized`` flag so repeated calls are no-ops.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._config = PurchasedGoodsServicesConfig()
            self._metrics = PurchasedGoodsServicesMetrics()
            self._decimal_places: int = self._config.decimal_places
            self._enable_transport: bool = self._config.enable_transport_adder
            self._enable_waste: bool = self._config.enable_waste_factor
            self._enable_dqi: bool = self._config.enable_dqi_scoring
            self._enable_provenance: bool = self._config.enable_provenance
            self._default_reporting_year: int = (
                self._config.default_reporting_year
            )
            # Provenance tracker (singleton via get_instance)
            try:
                self._provenance = (
                    PurchasedGoodsProvenanceTracker.get_instance()
                )
            except RuntimeError:
                self._provenance = None
                logger.warning(
                    "Provenance tracker unavailable; provenance "
                    "hashing will be computed inline"
                )
            self.__class__._initialized = True
            logger.info(
                "AverageDataCalculatorEngine initialised: "
                "agent=%s, version=%s, decimal_places=%d, "
                "transport_adder=%s, waste_factor=%s, "
                "dqi_scoring=%s, provenance=%s, "
                "ef_count=%d, unit_count=%d, transport_modes=%d",
                AGENT_ID,
                VERSION,
                self._decimal_places,
                self._enable_transport,
                self._enable_waste,
                self._enable_dqi,
                self._enable_provenance,
                len(PHYSICAL_EMISSION_FACTORS),
                len(UNIT_CONVERSION_TO_KG),
                len(TRANSPORT_EFS),
            )

    # ------------------------------------------------------------------
    # Singleton reset (for testing only)
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        Acquires the class lock and clears both ``_instance`` and
        ``_initialized``, allowing a fresh initialisation on next call.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.debug("AverageDataCalculatorEngine singleton reset")

    # ==================================================================
    # PUBLIC API -- Unit Conversion
    # ==================================================================

    def get_conversion_factor(self, unit: str) -> Optional[Decimal]:
        """Look up the conversion factor for a unit string.

        Performs case-insensitive lookup against
        ``UNIT_CONVERSION_TO_KG``.  Strips whitespace and normalises
        to lowercase before lookup.

        Args:
            unit: Unit string (e.g. ``"kg"``, ``"tonnes"``, ``"lb"``).

        Returns:
            Conversion factor as Decimal, or ``None`` if the unit is
            not recognised.

        Example:
            >>> engine = AverageDataCalculatorEngine()
            >>> engine.get_conversion_factor("tonnes")
            Decimal('1000')
            >>> engine.get_conversion_factor("unknown") is None
            True
        """
        if not unit:
            return None
        normalised = unit.strip().lower().replace(" ", "_")
        return UNIT_CONVERSION_TO_KG.get(normalised)

    def convert_to_kg(
        self,
        quantity: Decimal,
        unit: str,
        material_key: Optional[str] = None,
    ) -> Decimal:
        """Convert a physical quantity to kilograms.

        Handles three unit families:

        1. **Mass units** -- Direct multiplication by the conversion
           factor (e.g. ``tonnes`` -> ``* 1000``).
        2. **Volume units** -- Multiplied by the volume factor (to
           litres) then by material density (kg/L).  If no material
           density is found, defaults to 1.0 kg/L.
        3. **Count units** -- Multiplied by count factor then by
           per-piece weight from ``_DEFAULT_PIECE_WEIGHT_KG`` or
           defaults to 1.0 kg.

        Args:
            quantity: The numeric quantity to convert.
            unit: The unit of the quantity.
            material_key: Optional material key for density/weight
                lookup.

        Returns:
            Quantity in kilograms, quantized to ``DECIMAL_PLACES``.

        Raises:
            ValueError: If the unit is not recognised.

        Example:
            >>> engine = AverageDataCalculatorEngine()
            >>> engine.convert_to_kg(Decimal("5"), "tonnes")
            Decimal('5000.00000000')
            >>> engine.convert_to_kg(Decimal("100"), "lb")
            Decimal('45.35923700')
        """
        if quantity < ZERO:
            raise ValueError(
                f"Quantity must be non-negative, got {quantity}"
            )

        normalised_unit = unit.strip().lower().replace(" ", "_")
        factor = UNIT_CONVERSION_TO_KG.get(normalised_unit)
        if factor is None:
            raise ValueError(
                f"Unrecognised unit '{unit}'. Supported units: "
                f"{sorted(UNIT_CONVERSION_TO_KG.keys())}"
            )

        # Volume units -- apply material density
        if normalised_unit in _VOLUME_UNITS:
            density = self._get_material_density(material_key)
            result = quantity * factor * density
            return _q(result, self._decimal_places)

        # Count units -- apply per-piece weight
        if normalised_unit in _COUNT_UNITS:
            piece_weight = self._get_piece_weight(material_key)
            result = quantity * factor * piece_weight
            return _q(result, self._decimal_places)

        # Mass units -- direct conversion
        result = quantity * factor
        return _q(result, self._decimal_places)

    # ==================================================================
    # PUBLIC API -- Material Resolution
    # ==================================================================

    def resolve_material_key(
        self,
        item: ProcurementItem,
    ) -> Optional[str]:
        """Resolve a material emission factor key from a ProcurementItem.

        Resolution priority:

        1. ``item.metadata["material_key"]`` if present and in EF table.
        2. ``CATEGORY_DEFAULT_MATERIAL[item.material_category]`` if
           ``material_category`` is set.
        3. Keyword matching against ``item.description`` using
           ``DESCRIPTION_KEYWORD_MAP`` (longest match first).
        4. ``None`` if no resolution is possible.

        Args:
            item: The procurement item to resolve.

        Returns:
            Material key string (e.g. ``"steel_world_avg"``) or
            ``None`` if resolution fails.

        Example:
            >>> item = ProcurementItem(
            ...     description="HDPE pellets",
            ...     spend_amount=Decimal("1000"),
            ...     material_category=MaterialCategory.PLASTICS,
            ... )
            >>> engine.resolve_material_key(item)
            'hdpe'
        """
        # Priority 1: Explicit metadata
        if item.metadata:
            explicit_key = item.metadata.get("material_key")
            if explicit_key and explicit_key in PHYSICAL_EMISSION_FACTORS:
                return str(explicit_key)

        # Priority 2: Category default
        if item.material_category is not None:
            cat_key = CATEGORY_DEFAULT_MATERIAL.get(item.material_category)
            if cat_key and cat_key in PHYSICAL_EMISSION_FACTORS:
                # But first check description for a more specific match
                desc_key = self._resolve_from_description(item.description)
                if desc_key is not None:
                    # Verify category consistency
                    desc_cat = MATERIAL_CATEGORY_MAP.get(desc_key)
                    if desc_cat == item.material_category:
                        return desc_key
                return cat_key

        # Priority 3: Description keywords
        desc_key = self._resolve_from_description(item.description)
        if desc_key is not None:
            return desc_key

        return None

    def resolve_ef_for_material(
        self,
        material_key: str,
        source: PhysicalEFSource = PhysicalEFSource.DEFRA,
    ) -> Optional[Decimal]:
        """Resolve the cradle-to-gate emission factor for a material.

        Looks up ``material_key`` in ``PHYSICAL_EMISSION_FACTORS``.
        The ``source`` parameter is recorded for provenance but does
        not currently filter the factor (all factors are sourced from
        the embedded constant table regardless of ``source``).

        Args:
            material_key: Material key string.
            source: EF source database (for provenance tracking).

        Returns:
            Emission factor in kgCO2e per kg, or ``None`` if the
            material key is not found.

        Example:
            >>> engine.resolve_ef_for_material("steel_world_avg")
            Decimal('1.37')
        """
        ef = PHYSICAL_EMISSION_FACTORS.get(material_key)
        if ef is not None:
            logger.debug(
                "Resolved EF for %s (source=%s): %s kgCO2e/kg",
                material_key,
                source.value,
                ef,
            )
        return ef

    # ==================================================================
    # PUBLIC API -- Transport Emissions
    # ==================================================================

    def get_transport_ef(self, mode: str) -> Decimal:
        """Get the transport emission factor for a given mode.

        Args:
            mode: Transport mode string (e.g. ``"road"``, ``"sea"``).

        Returns:
            Emission factor in kgCO2e per tonne-km.

        Raises:
            ValueError: If the transport mode is not recognised.

        Example:
            >>> engine.get_transport_ef("road")
            Decimal('0.0620')
        """
        normalised = mode.strip().lower().replace(" ", "_")
        ef = TRANSPORT_EFS.get(normalised)
        if ef is None:
            raise ValueError(
                f"Unrecognised transport mode '{mode}'. "
                f"Supported modes: {sorted(TRANSPORT_EFS.keys())}"
            )
        return ef

    def calculate_transport_emissions(
        self,
        quantity_kg: Decimal,
        distance_km: Decimal,
        mode: str,
    ) -> Decimal:
        """Calculate transport-to-gate emissions.

        Formula::

            Transport_kgCO2e = (Qty_kg / 1000) x Distance_km x EF_tkm

        The quantity is divided by 1000 to convert kg to tonnes before
        multiplying by the distance and the tonne-km emission factor.

        Args:
            quantity_kg: Quantity of goods in kilograms.
            distance_km: Transport distance in kilometres.
            mode: Transport mode for EF lookup.

        Returns:
            Transport emissions in kgCO2e, quantized.

        Raises:
            ValueError: If mode is not recognised or inputs are
                negative.

        Example:
            >>> engine.calculate_transport_emissions(
            ...     Decimal("10000"),
            ...     Decimal("500"),
            ...     "road",
            ... )
            Decimal('310.00000000')
        """
        if quantity_kg < ZERO:
            raise ValueError(
                f"quantity_kg must be non-negative, got {quantity_kg}"
            )
        if distance_km < ZERO:
            raise ValueError(
                f"distance_km must be non-negative, got {distance_km}"
            )
        if quantity_kg == ZERO or distance_km == ZERO:
            return _q(ZERO, self._decimal_places)

        ef_tkm = self.get_transport_ef(mode)
        # Convert kg to tonnes
        quantity_tonnes = _q(
            quantity_kg / ONE_THOUSAND, self._decimal_places
        )
        transport = _q(
            quantity_tonnes * distance_km * ef_tkm,
            self._decimal_places,
        )
        logger.debug(
            "Transport emissions: %.4f t x %.1f km x %.4f EF = %.4f kgCO2e "
            "(mode=%s)",
            quantity_tonnes,
            distance_km,
            ef_tkm,
            transport,
            mode,
        )
        return transport

    # ==================================================================
    # PUBLIC API -- Single Item Calculation
    # ==================================================================

    def calculate_single(
        self,
        item: ProcurementItem,
        material_key: Optional[str] = None,
        ef_source: PhysicalEFSource = PhysicalEFSource.DEFRA,
        includes_transport: bool = True,
        transport_distance_km: Decimal = ZERO,
        transport_mode: str = "road",
        waste_loss_factor: Decimal = ONE,
    ) -> AverageDataResult:
        """Calculate average-data emissions for a single procurement item.

        This is the primary entry point for average-data calculations.
        The method:

        1. Resolves the material key (explicit or via resolution).
        2. Converts the item quantity to kilograms.
        3. Looks up the cradle-to-gate emission factor.
        4. Applies the core formula with waste/loss factor.
        5. Optionally adds transport emissions.
        6. Computes SHA-256 provenance hash.
        7. Records metrics.

        Args:
            item: The procurement item with quantity and unit data.
            material_key: Explicit material key; if None, resolved
                from item attributes.
            ef_source: Source database for the emission factor.
            includes_transport: Whether the EF includes transport to
                gate.  If False and transport distance > 0, a transport
                adder is calculated.
            transport_distance_km: Distance in km for transport adder.
            transport_mode: Transport mode for transport adder.
            waste_loss_factor: Multiplicative waste/loss factor
                (>= 1.0, default 1.0 = no loss).

        Returns:
            AverageDataResult with calculated emissions and metadata.

        Raises:
            ValueError: If the item has no quantity or unit, material
                key cannot be resolved, emission factor not found, or
                waste_loss_factor < 1.0.

        Example:
            >>> item = ProcurementItem(
            ...     description="HDPE pellets",
            ...     spend_amount=Decimal("25000"),
            ...     quantity=Decimal("5000"),
            ...     quantity_unit="kg",
            ... )
            >>> result = engine.calculate_single(
            ...     item,
            ...     material_key="hdpe",
            ... )
            >>> result.emissions_kgco2e
            Decimal('9000.00000000')
        """
        start_time = time.monotonic()

        # -- Validate inputs --
        self._validate_item_for_average_data(item)
        if waste_loss_factor < ONE:
            raise ValueError(
                f"waste_loss_factor must be >= 1.0, got {waste_loss_factor}"
            )

        # -- Resolve material key --
        resolved_key = material_key
        if resolved_key is None:
            resolved_key = self.resolve_material_key(item)
        if resolved_key is None:
            raise ValueError(
                f"Cannot resolve material key for item "
                f"'{item.item_id}' ({item.description}). "
                f"Provide material_key explicitly or set "
                f"material_category on the item."
            )

        # -- Look up emission factor --
        ef_kgco2e = self.resolve_ef_for_material(resolved_key, ef_source)
        if ef_kgco2e is None:
            raise ValueError(
                f"No emission factor found for material_key "
                f"'{resolved_key}' (source={ef_source.value}). "
                f"Available keys: {sorted(PHYSICAL_EMISSION_FACTORS.keys())}"
            )

        # -- Convert quantity to kg --
        quantity_kg = self.convert_to_kg(
            item.quantity,  # type: ignore[arg-type]
            item.quantity_unit,  # type: ignore[arg-type]
            resolved_key,
        )

        # -- Core calculation --
        effective_waste = (
            waste_loss_factor if self._enable_waste else ONE
        )
        base_emissions = _q(
            quantity_kg * ef_kgco2e * effective_waste,
            self._decimal_places,
        )

        # -- Transport adder --
        transport_emissions = _q(ZERO, self._decimal_places)
        if (
            self._enable_transport
            and not includes_transport
            and transport_distance_km > ZERO
        ):
            transport_emissions = self.calculate_transport_emissions(
                quantity_kg, transport_distance_km, transport_mode
            )

        total_with_transport = _q(
            base_emissions + transport_emissions,
            self._decimal_places,
        )
        emissions_tco2e = _q(
            total_with_transport / ONE_THOUSAND,
            self._decimal_places,
        )

        # -- Provenance hash --
        provenance_hash = self._compute_result_provenance(
            item_id=item.item_id,
            material_key=resolved_key,
            quantity_kg=quantity_kg,
            ef_kgco2e=ef_kgco2e,
            waste_loss_factor=effective_waste,
            base_emissions=base_emissions,
            transport_emissions=transport_emissions,
            total_emissions=total_with_transport,
        )

        # -- Build result --
        result = AverageDataResult(
            item_id=item.item_id,
            emissions_kgco2e=base_emissions,
            emissions_tco2e=emissions_tco2e,
            quantity_kg=quantity_kg,
            ef_kgco2e_per_kg=ef_kgco2e,
            ef_source=ef_source,
            material_key=resolved_key,
            transport_emissions_kgco2e=transport_emissions,
            waste_loss_factor=effective_waste,
            total_with_transport_kgco2e=total_with_transport,
            provenance_hash=provenance_hash,
        )

        # -- Metrics --
        duration_s = time.monotonic() - start_time
        self._record_calculation_metric(
            item=item,
            result=result,
            duration_s=duration_s,
            status="success",
        )

        logger.info(
            "Average-data calc: item=%s, material=%s, qty=%.2f kg, "
            "EF=%.4f, waste=%.2f, base=%.4f, transport=%.4f, "
            "total=%.4f kgCO2e (%.6f tCO2e) [%.3f ms]",
            item.item_id,
            resolved_key,
            quantity_kg,
            ef_kgco2e,
            effective_waste,
            base_emissions,
            transport_emissions,
            total_with_transport,
            emissions_tco2e,
            duration_s * 1000,
        )

        return result

    # ==================================================================
    # PUBLIC API -- Record-Based Calculation
    # ==================================================================

    def calculate_from_record(
        self,
        record: PhysicalRecord,
    ) -> AverageDataResult:
        """Calculate emissions from a pre-enriched PhysicalRecord.

        A ``PhysicalRecord`` is a ``ProcurementItem`` enriched with
        quantity-in-kg, material key, EF source, transport parameters,
        and waste/loss factor.  This method uses the record's fields
        directly, bypassing material resolution and unit conversion
        (since ``quantity_kg`` is already normalised).

        If ``record.quantity_kg`` is zero but the underlying item has
        a non-zero quantity and unit, the engine will convert from the
        item's quantity/unit to populate ``quantity_kg``.

        Args:
            record: Pre-enriched PhysicalRecord with all fields
                populated.

        Returns:
            AverageDataResult with calculated emissions.

        Raises:
            ValueError: If material_key is missing or EF not found.

        Example:
            >>> record = PhysicalRecord(
            ...     item=item,
            ...     quantity_kg=Decimal("5000"),
            ...     material_key="hdpe",
            ...     ef_source=PhysicalEFSource.DEFRA,
            ... )
            >>> result = engine.calculate_from_record(record)
        """
        start_time = time.monotonic()

        # Resolve quantity_kg if not pre-populated
        quantity_kg = record.quantity_kg
        if quantity_kg == ZERO and record.item.quantity and record.item.quantity_unit:
            quantity_kg = self.convert_to_kg(
                record.item.quantity,
                record.item.quantity_unit,
                record.material_key,
            )

        # Resolve material key
        material_key = record.material_key
        if material_key is None:
            material_key = self.resolve_material_key(record.item)
        if material_key is None:
            raise ValueError(
                f"Cannot resolve material key for record item "
                f"'{record.item.item_id}'"
            )

        # Look up emission factor
        ef_kgco2e = self.resolve_ef_for_material(
            material_key, record.ef_source
        )
        if ef_kgco2e is None:
            raise ValueError(
                f"No emission factor for material '{material_key}'"
            )

        # Core calculation
        effective_waste = (
            record.waste_loss_factor if self._enable_waste else ONE
        )
        base_emissions = _q(
            quantity_kg * ef_kgco2e * effective_waste,
            self._decimal_places,
        )

        # Transport adder
        transport_emissions = _q(ZERO, self._decimal_places)
        if (
            self._enable_transport
            and not record.includes_transport
            and record.transport_distance_km > ZERO
            and record.transport_mode
        ):
            transport_emissions = self.calculate_transport_emissions(
                quantity_kg,
                record.transport_distance_km,
                record.transport_mode,
            )

        total_with_transport = _q(
            base_emissions + transport_emissions,
            self._decimal_places,
        )
        emissions_tco2e = _q(
            total_with_transport / ONE_THOUSAND,
            self._decimal_places,
        )

        # Provenance
        provenance_hash = self._compute_result_provenance(
            item_id=record.item.item_id,
            material_key=material_key,
            quantity_kg=quantity_kg,
            ef_kgco2e=ef_kgco2e,
            waste_loss_factor=effective_waste,
            base_emissions=base_emissions,
            transport_emissions=transport_emissions,
            total_emissions=total_with_transport,
        )

        result = AverageDataResult(
            item_id=record.item.item_id,
            emissions_kgco2e=base_emissions,
            emissions_tco2e=emissions_tco2e,
            quantity_kg=quantity_kg,
            ef_kgco2e_per_kg=ef_kgco2e,
            ef_source=record.ef_source,
            material_key=material_key,
            transport_emissions_kgco2e=transport_emissions,
            waste_loss_factor=effective_waste,
            total_with_transport_kgco2e=total_with_transport,
            provenance_hash=provenance_hash,
        )

        duration_s = time.monotonic() - start_time
        self._record_calculation_metric(
            item=record.item,
            result=result,
            duration_s=duration_s,
            status="success",
        )

        logger.debug(
            "Record-based avg-data calc: item=%s, material=%s, "
            "total=%.4f kgCO2e [%.3f ms]",
            record.item.item_id,
            material_key,
            total_with_transport,
            duration_s * 1000,
        )

        return result

    # ==================================================================
    # PUBLIC API -- Batch Processing
    # ==================================================================

    def calculate_batch(
        self,
        items: List[ProcurementItem],
    ) -> List[AverageDataResult]:
        """Calculate average-data emissions for a batch of items.

        Processes each item independently with error isolation.  Items
        that fail (e.g. missing quantity, unresolvable material) are
        skipped with a WARNING log.  The returned list contains only
        successful results, in the same order as the input items that
        succeeded.

        Args:
            items: List of ProcurementItem objects.

        Returns:
            List of AverageDataResult for successfully processed items.

        Example:
            >>> results = engine.calculate_batch([item1, item2, item3])
            >>> len(results)  # may be < 3 if some items failed
            2
        """
        start_time = time.monotonic()
        results: List[AverageDataResult] = []
        success_count = 0
        failure_count = 0
        skip_count = 0

        for idx, item in enumerate(items):
            # Skip items without quantity data
            if item.quantity is None or item.quantity_unit is None:
                logger.warning(
                    "Skipping item %d/%d (%s): missing quantity or "
                    "quantity_unit for average-data method",
                    idx + 1,
                    len(items),
                    item.item_id,
                )
                skip_count += 1
                continue

            if item.quantity <= ZERO:
                logger.warning(
                    "Skipping item %d/%d (%s): quantity is zero or "
                    "negative",
                    idx + 1,
                    len(items),
                    item.item_id,
                )
                skip_count += 1
                continue

            try:
                result = self.calculate_single(item)
                results.append(result)
                success_count += 1
            except (ValueError, InvalidOperation) as exc:
                logger.warning(
                    "Failed to calculate item %d/%d (%s): %s",
                    idx + 1,
                    len(items),
                    item.item_id,
                    str(exc),
                )
                failure_count += 1

        duration_s = time.monotonic() - start_time
        logger.info(
            "Batch avg-data calculation complete: %d items, "
            "%d success, %d failed, %d skipped [%.3f s]",
            len(items),
            success_count,
            failure_count,
            skip_count,
            duration_s,
        )

        return results

    def calculate_batch_records(
        self,
        records: List[PhysicalRecord],
    ) -> List[AverageDataResult]:
        """Calculate emissions for a batch of pre-enriched PhysicalRecords.

        Like ``calculate_batch`` but operates on pre-enriched records
        instead of raw ``ProcurementItem`` objects.  Each record is
        processed independently with error isolation.

        Args:
            records: List of PhysicalRecord objects.

        Returns:
            List of AverageDataResult for successfully processed
            records.

        Example:
            >>> results = engine.calculate_batch_records([rec1, rec2])
        """
        start_time = time.monotonic()
        results: List[AverageDataResult] = []
        success_count = 0
        failure_count = 0

        for idx, record in enumerate(records):
            try:
                result = self.calculate_from_record(record)
                results.append(result)
                success_count += 1
            except (ValueError, InvalidOperation) as exc:
                logger.warning(
                    "Failed to calculate record %d/%d (%s): %s",
                    idx + 1,
                    len(records),
                    record.item.item_id,
                    str(exc),
                )
                failure_count += 1

        duration_s = time.monotonic() - start_time
        logger.info(
            "Batch record avg-data calculation: %d records, "
            "%d success, %d failed [%.3f s]",
            len(records),
            success_count,
            failure_count,
            duration_s,
        )

        return results

    # ==================================================================
    # PUBLIC API -- Multi-Material BOM Calculation
    # ==================================================================

    def calculate_multi_material(
        self,
        materials: List[Dict[str, Any]],
    ) -> AverageDataResult:
        """Calculate emissions for a multi-material (BOM) product.

        Accepts a bill-of-materials (BOM) list where each entry
        specifies a material key, quantity, unit, and optional
        transport parameters.  Each material component is calculated
        independently and the results are aggregated into a single
        ``AverageDataResult``.

        Each material dict must contain:
            - ``material_key`` (str): Material EF lookup key.
            - ``quantity`` (Decimal or str): Quantity of this material.
            - ``unit`` (str): Unit of the quantity.

        Optional fields:
            - ``ef_source`` (str): PhysicalEFSource value.
            - ``includes_transport`` (bool): Default True.
            - ``transport_distance_km`` (Decimal or str): Default 0.
            - ``transport_mode`` (str): Default "road".
            - ``waste_loss_factor`` (Decimal or str): Default 1.0.
            - ``weight_fraction`` (Decimal or str): Mass fraction in
              the final product (0-1).  If provided, the component
              emission is weighted by this fraction.

        Args:
            materials: List of material component dictionaries.

        Returns:
            Aggregated AverageDataResult for the entire product.

        Raises:
            ValueError: If materials list is empty or any component
                fails validation.

        Example:
            >>> materials = [
            ...     {
            ...         "material_key": "steel_world_avg",
            ...         "quantity": "100",
            ...         "unit": "kg",
            ...         "weight_fraction": "0.60",
            ...     },
            ...     {
            ...         "material_key": "hdpe",
            ...         "quantity": "40",
            ...         "unit": "kg",
            ...         "weight_fraction": "0.25",
            ...     },
            ...     {
            ...         "material_key": "natural_rubber",
            ...         "quantity": "25",
            ...         "unit": "kg",
            ...         "weight_fraction": "0.15",
            ...     },
            ... ]
            >>> result = engine.calculate_multi_material(materials)
        """
        start_time = time.monotonic()

        if not materials:
            raise ValueError("materials list must not be empty")

        total_emissions_kgco2e = _q(ZERO, self._decimal_places)
        total_transport_kgco2e = _q(ZERO, self._decimal_places)
        total_quantity_kg = _q(ZERO, self._decimal_places)
        weighted_ef_sum = _q(ZERO, self._decimal_places)
        weighted_waste_sum = _q(ZERO, self._decimal_places)
        component_count = 0
        material_keys_used: List[str] = []
        provenance_components: List[Dict[str, str]] = []

        # Synthesise a composite item_id
        composite_id = f"bom-{uuid.uuid4().hex[:12]}"

        for idx, mat in enumerate(materials):
            # -- Parse fields --
            mat_key = str(mat.get("material_key", ""))
            if not mat_key:
                raise ValueError(
                    f"Material {idx}: 'material_key' is required"
                )

            try:
                qty = Decimal(str(mat.get("quantity", "0")))
            except InvalidOperation:
                raise ValueError(
                    f"Material {idx}: invalid quantity "
                    f"'{mat.get('quantity')}'"
                )

            unit_str = str(mat.get("unit", "kg"))

            ef_source_str = str(
                mat.get("ef_source", PhysicalEFSource.DEFRA.value)
            )
            try:
                ef_source = PhysicalEFSource(ef_source_str)
            except ValueError:
                ef_source = PhysicalEFSource.DEFRA

            includes_transport = bool(
                mat.get("includes_transport", True)
            )

            try:
                transport_dist = Decimal(
                    str(mat.get("transport_distance_km", "0"))
                )
            except InvalidOperation:
                transport_dist = ZERO

            transport_mode = str(mat.get("transport_mode", "road"))

            try:
                waste = Decimal(
                    str(mat.get("waste_loss_factor", "1.0"))
                )
            except InvalidOperation:
                waste = ONE
            if waste < ONE:
                waste = ONE

            try:
                weight_frac = Decimal(
                    str(mat.get("weight_fraction", "1.0"))
                )
            except InvalidOperation:
                weight_frac = ONE

            # -- Look up EF --
            ef_kgco2e = self.resolve_ef_for_material(mat_key, ef_source)
            if ef_kgco2e is None:
                raise ValueError(
                    f"Material {idx}: no EF for key '{mat_key}'"
                )

            # -- Convert to kg --
            qty_kg = self.convert_to_kg(qty, unit_str, mat_key)

            # -- Core calculation --
            effective_waste = waste if self._enable_waste else ONE
            component_emissions = _q(
                qty_kg * ef_kgco2e * effective_waste * weight_frac,
                self._decimal_places,
            )

            # -- Transport adder --
            component_transport = _q(ZERO, self._decimal_places)
            if (
                self._enable_transport
                and not includes_transport
                and transport_dist > ZERO
            ):
                raw_transport = self.calculate_transport_emissions(
                    qty_kg, transport_dist, transport_mode
                )
                component_transport = _q(
                    raw_transport * weight_frac,
                    self._decimal_places,
                )

            # -- Accumulate --
            total_emissions_kgco2e += component_emissions
            total_transport_kgco2e += component_transport
            total_quantity_kg += _q(
                qty_kg * weight_frac, self._decimal_places
            )
            weighted_ef_sum += _q(
                ef_kgco2e * weight_frac, self._decimal_places
            )
            weighted_waste_sum += _q(
                effective_waste * weight_frac, self._decimal_places
            )
            component_count += 1
            material_keys_used.append(mat_key)
            provenance_components.append({
                "material_key": mat_key,
                "quantity_kg": str(qty_kg),
                "ef": str(ef_kgco2e),
                "emissions": str(component_emissions),
            })

            logger.debug(
                "BOM component %d: %s qty=%.2f kg, EF=%.4f, "
                "frac=%.2f, emissions=%.4f kgCO2e",
                idx,
                mat_key,
                qty_kg,
                ef_kgco2e,
                weight_frac,
                component_emissions,
            )

        # -- Aggregate --
        total_emissions_kgco2e = _q(
            total_emissions_kgco2e, self._decimal_places
        )
        total_transport_kgco2e = _q(
            total_transport_kgco2e, self._decimal_places
        )
        total_with_transport = _q(
            total_emissions_kgco2e + total_transport_kgco2e,
            self._decimal_places,
        )
        emissions_tco2e = _q(
            total_with_transport / ONE_THOUSAND,
            self._decimal_places,
        )

        # Weighted average EF
        avg_ef = (
            _q(weighted_ef_sum / Decimal(str(component_count)),
               self._decimal_places)
            if component_count > 0
            else ZERO
        )
        avg_waste = (
            _q(weighted_waste_sum / Decimal(str(component_count)),
               self._decimal_places)
            if component_count > 0
            else ONE
        )

        # Primary material (largest emission contributor)
        primary_material = (
            material_keys_used[0] if material_keys_used else "unknown"
        )

        # Provenance
        provenance_hash = _sha256({
            "engine": "average_data_calculator",
            "method": "multi_material",
            "composite_id": composite_id,
            "component_count": component_count,
            "components": provenance_components,
            "total_emissions_kgco2e": str(total_emissions_kgco2e),
            "total_transport_kgco2e": str(total_transport_kgco2e),
        })

        result = AverageDataResult(
            item_id=composite_id,
            emissions_kgco2e=total_emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            quantity_kg=total_quantity_kg,
            ef_kgco2e_per_kg=avg_ef,
            ef_source=PhysicalEFSource.DEFRA,
            material_key=primary_material,
            transport_emissions_kgco2e=total_transport_kgco2e,
            waste_loss_factor=avg_waste,
            total_with_transport_kgco2e=total_with_transport,
            provenance_hash=provenance_hash,
        )

        duration_s = time.monotonic() - start_time
        logger.info(
            "Multi-material BOM calc: %d components, "
            "total=%.4f kgCO2e, transport=%.4f kgCO2e [%.3f ms]",
            component_count,
            total_emissions_kgco2e,
            total_transport_kgco2e,
            duration_s * 1000,
        )

        return result

    # ==================================================================
    # PUBLIC API -- DQI Scoring
    # ==================================================================

    def score_dqi_average_data(
        self,
        item: ProcurementItem,
        ef_source: PhysicalEFSource,
        result: AverageDataResult,
    ) -> DQIAssessment:
        """Score data quality for an average-data calculation result.

        Evaluates the five GHG Protocol DQI dimensions:

        1. **Temporal** -- How recent is the emission factor relative
           to the reporting period?
        2. **Geographical** -- How well does the EF's geographic scope
           match the item's procurement geography?
        3. **Technological** -- How well does the EF's technology scope
           match the item's actual production process?
        4. **Completeness** -- Are all relevant emission sources
           included in the calculation?
        5. **Reliability** -- Is the data source trustworthy and the
           methodology sound?

        Scoring rules for average-data method:

        - Temporal: Score 2 (factors updated within 3 years)
        - Geographical: Score 3 (global/regional averages)
        - Technological: Score 3 (industry average technology)
        - Completeness: Score 2 if transport included, else 3
        - Reliability: Score 2 for recognised sources (ecoinvent,
          World Steel, ICE), 3 for DEFRA, 4 for custom

        The composite score is the arithmetic mean of all five.  The
        quality tier is determined from the composite score using
        ``DQI_QUALITY_TIERS``.

        Args:
            item: The procurement item.
            ef_source: The EF source used.
            result: The calculation result to score.

        Returns:
            DQIAssessment with dimension scores, composite, and tier.

        Example:
            >>> dqi = engine.score_dqi_average_data(item, ef_source, result)
            >>> dqi.composite_score
            Decimal('2.6')
        """
        findings: List[str] = []

        # -- Temporal score --
        temporal = self._score_temporal_average_data(item, ef_source)
        if temporal > Decimal("2.0"):
            findings.append(
                f"Temporal score {temporal}: EF source ({ef_source.value}) "
                f"may not reflect current reporting period"
            )

        # -- Geographical score --
        geographical = self._score_geographical_average_data(
            item, ef_source
        )
        if geographical > Decimal("2.0"):
            findings.append(
                f"Geographical score {geographical}: EF is a "
                f"global/regional average, not country-specific"
            )

        # -- Technological score --
        technological = self._score_technological_average_data(
            item, result
        )
        if technological > Decimal("2.0"):
            findings.append(
                f"Technological score {technological}: industry-average "
                f"EF may not match actual production technology"
            )

        # -- Completeness score --
        completeness = self._score_completeness_average_data(result)
        if completeness > Decimal("2.0"):
            findings.append(
                f"Completeness score {completeness}: "
                f"{'transport excluded from EF' if result.transport_emissions_kgco2e == ZERO and not result.waste_loss_factor > ONE else 'some sources may be missing'}"
            )

        # -- Reliability score --
        reliability = self._score_reliability_average_data(ef_source)
        if reliability > Decimal("2.0"):
            findings.append(
                f"Reliability score {reliability}: source "
                f"({ef_source.value}) is not third-party verified"
            )

        # -- Composite --
        five = Decimal("5")
        composite = _q(
            (temporal + geographical + technological
             + completeness + reliability) / five,
            self._decimal_places,
        )

        # -- Quality tier --
        quality_tier = self._determine_quality_tier(composite)

        # -- EF hierarchy level for average data --
        ef_hierarchy = self._determine_ef_hierarchy_level(ef_source)

        # -- Uncertainty factor from pedigree --
        uncertainty_factor = self._compute_pedigree_uncertainty(
            composite
        )

        # -- Recommendations --
        if composite > Decimal("3.0"):
            findings.append(
                "RECOMMENDATION: Consider obtaining supplier-specific "
                "data (EPD/PCF) to improve data quality"
            )
        if composite > Decimal("2.5"):
            findings.append(
                "RECOMMENDATION: Use product-level LCA data from "
                "ecoinvent or GaBi for higher accuracy"
            )

        dqi = DQIAssessment(
            item_id=item.item_id,
            calculation_method=CalculationMethod.AVERAGE_DATA,
            temporal_score=temporal,
            geographical_score=geographical,
            technological_score=technological,
            completeness_score=completeness,
            reliability_score=reliability,
            composite_score=composite,
            quality_tier=quality_tier,
            uncertainty_factor=uncertainty_factor,
            findings=findings,
            ef_hierarchy_level=ef_hierarchy,
        )

        logger.debug(
            "DQI scored for %s: T=%.1f G=%.1f Te=%.1f C=%.1f R=%.1f "
            "composite=%.2f tier=%s",
            item.item_id,
            temporal,
            geographical,
            technological,
            completeness,
            reliability,
            composite,
            quality_tier,
        )

        return dqi

    # ==================================================================
    # PUBLIC API -- Aggregation
    # ==================================================================

    def aggregate_by_material(
        self,
        results: List[AverageDataResult],
    ) -> Dict[str, Dict[str, Decimal]]:
        """Aggregate calculation results by material key.

        Groups results by ``material_key`` and computes per-material
        totals for emissions, transport, and quantity.

        Args:
            results: List of AverageDataResult objects.

        Returns:
            Dictionary keyed by material_key, each containing:
                - ``total_emissions_kgco2e``: Sum of base emissions.
                - ``total_transport_kgco2e``: Sum of transport adder.
                - ``total_with_transport_kgco2e``: Grand total.
                - ``total_quantity_kg``: Sum of quantities.
                - ``avg_ef_kgco2e_per_kg``: Average emission factor.
                - ``item_count``: Number of items.
                - ``total_emissions_tco2e``: Sum in tonnes.

        Example:
            >>> agg = engine.aggregate_by_material(results)
            >>> agg["steel_world_avg"]["total_emissions_kgco2e"]
            Decimal('54800.00000000')
        """
        aggregation: Dict[str, Dict[str, Decimal]] = {}

        for result in results:
            key = result.material_key
            if key not in aggregation:
                aggregation[key] = {
                    "total_emissions_kgco2e": _q(ZERO, self._decimal_places),
                    "total_transport_kgco2e": _q(ZERO, self._decimal_places),
                    "total_with_transport_kgco2e": _q(
                        ZERO, self._decimal_places
                    ),
                    "total_quantity_kg": _q(ZERO, self._decimal_places),
                    "ef_sum": _q(ZERO, self._decimal_places),
                    "item_count": ZERO,
                    "total_emissions_tco2e": _q(ZERO, self._decimal_places),
                }

            entry = aggregation[key]
            entry["total_emissions_kgco2e"] = _q(
                entry["total_emissions_kgco2e"] + result.emissions_kgco2e,
                self._decimal_places,
            )
            entry["total_transport_kgco2e"] = _q(
                entry["total_transport_kgco2e"]
                + result.transport_emissions_kgco2e,
                self._decimal_places,
            )
            entry["total_with_transport_kgco2e"] = _q(
                entry["total_with_transport_kgco2e"]
                + result.total_with_transport_kgco2e,
                self._decimal_places,
            )
            entry["total_quantity_kg"] = _q(
                entry["total_quantity_kg"] + result.quantity_kg,
                self._decimal_places,
            )
            entry["ef_sum"] = _q(
                entry["ef_sum"] + result.ef_kgco2e_per_kg,
                self._decimal_places,
            )
            entry["item_count"] += ONE
            entry["total_emissions_tco2e"] = _q(
                entry["total_emissions_tco2e"] + result.emissions_tco2e,
                self._decimal_places,
            )

        # Compute average EF and remove internal accumulator
        for key, entry in aggregation.items():
            count = entry["item_count"]
            if count > ZERO:
                entry["avg_ef_kgco2e_per_kg"] = _q(
                    entry["ef_sum"] / count,
                    self._decimal_places,
                )
            else:
                entry["avg_ef_kgco2e_per_kg"] = _q(
                    ZERO, self._decimal_places
                )
            del entry["ef_sum"]

        logger.debug(
            "Aggregated %d results into %d material groups",
            len(results),
            len(aggregation),
        )

        return aggregation

    # ==================================================================
    # PUBLIC API -- Coverage
    # ==================================================================

    def compute_coverage(
        self,
        results: List[AverageDataResult],
        total_items: int,
    ) -> Dict[str, Decimal]:
        """Compute coverage statistics for a batch calculation.

        Reports the number and percentage of items successfully
        calculated relative to the total number of items attempted.

        Args:
            results: List of successful AverageDataResult objects.
            total_items: Total number of items in the original batch.

        Returns:
            Dictionary containing:
                - ``calculated_count``: Number of items calculated.
                - ``total_count``: Total items in batch.
                - ``coverage_pct``: Percentage covered (0-100).
                - ``missing_count``: Items not calculated.
                - ``missing_pct``: Percentage missing.
                - ``total_emissions_kgco2e``: Sum of emissions.
                - ``total_emissions_tco2e``: Sum in tonnes.

        Raises:
            ValueError: If total_items < 0.

        Example:
            >>> cov = engine.compute_coverage(results, 100)
            >>> cov["coverage_pct"]
            Decimal('85.00000000')
        """
        if total_items < 0:
            raise ValueError(
                f"total_items must be non-negative, got {total_items}"
            )

        calc_count = Decimal(str(len(results)))
        total_count = Decimal(str(total_items))

        if total_count == ZERO:
            return {
                "calculated_count": ZERO,
                "total_count": ZERO,
                "coverage_pct": _q(ZERO, self._decimal_places),
                "missing_count": ZERO,
                "missing_pct": _q(ZERO, self._decimal_places),
                "total_emissions_kgco2e": _q(ZERO, self._decimal_places),
                "total_emissions_tco2e": _q(ZERO, self._decimal_places),
            }

        coverage_pct = _q(
            (calc_count / total_count) * ONE_HUNDRED,
            self._decimal_places,
        )
        missing_count = total_count - calc_count
        missing_pct = _q(
            (missing_count / total_count) * ONE_HUNDRED,
            self._decimal_places,
        )

        total_emissions = _q(ZERO, self._decimal_places)
        total_tco2e = _q(ZERO, self._decimal_places)
        for r in results:
            total_emissions = _q(
                total_emissions + r.total_with_transport_kgco2e,
                self._decimal_places,
            )
            total_tco2e = _q(
                total_tco2e + r.emissions_tco2e,
                self._decimal_places,
            )

        return {
            "calculated_count": calc_count,
            "total_count": total_count,
            "coverage_pct": coverage_pct,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_tco2e,
        }

    # ==================================================================
    # PUBLIC API -- Uncertainty Estimation
    # ==================================================================

    def estimate_uncertainty(
        self,
        result: AverageDataResult,
    ) -> Dict[str, Decimal]:
        """Estimate uncertainty range for an average-data result.

        Combines:

        1. **Method-level uncertainty** from ``UNCERTAINTY_RANGES``
           for ``AVERAGE_DATA`` (+/- 30-60%).
        2. **Material-specific uncertainty** from
           ``MATERIAL_UNCERTAINTY_PCT`` if available.
        3. **Pedigree factor** from the material's known quality.

        Returns a symmetric confidence interval at 95% confidence.

        Args:
            result: The AverageDataResult to assess.

        Returns:
            Dictionary containing:
                - ``emissions_kgco2e``: Central estimate.
                - ``uncertainty_pct_lower``: Lower bound %.
                - ``uncertainty_pct_upper``: Upper bound %.
                - ``lower_bound_kgco2e``: Lower bound emissions.
                - ``upper_bound_kgco2e``: Upper bound emissions.
                - ``method``: "pedigree_analytical".
                - ``confidence_level_pct``: Decimal("95.0").
                - ``material_uncertainty_pct``: Material-specific %.

        Example:
            >>> unc = engine.estimate_uncertainty(result)
            >>> unc["uncertainty_pct_lower"]
            Decimal('-45.00000000')
        """
        # Method-level range
        method_range = UNCERTAINTY_RANGES.get(
            CalculationMethod.AVERAGE_DATA,
            (Decimal("30"), Decimal("60")),
        )
        method_lower = method_range[0]
        method_upper = method_range[1]

        # Material-specific uncertainty
        material_unc = MATERIAL_UNCERTAINTY_PCT.get(
            result.material_key, Decimal("40")
        )

        # Combined uncertainty via root-sum-of-squares (analytical)
        # Combined = sqrt(method_mid^2 + material^2)
        method_mid = _q(
            (method_lower + method_upper) / Decimal("2"),
            self._decimal_places,
        )

        # Use simplified analytical propagation
        combined_sq = (method_mid * method_mid) + (material_unc * material_unc)
        # Approximate sqrt using Newton's method for Decimal
        combined_unc = self._decimal_sqrt(combined_sq)

        lower_pct = _q(-combined_unc, self._decimal_places)
        upper_pct = _q(combined_unc, self._decimal_places)

        emissions = result.total_with_transport_kgco2e
        lower_bound = _q(
            emissions * (ONE + lower_pct / ONE_HUNDRED),
            self._decimal_places,
        )
        upper_bound = _q(
            emissions * (ONE + upper_pct / ONE_HUNDRED),
            self._decimal_places,
        )

        # Clamp lower bound to zero
        if lower_bound < ZERO:
            lower_bound = _q(ZERO, self._decimal_places)

        return {
            "emissions_kgco2e": _q(emissions, self._decimal_places),
            "uncertainty_pct_lower": lower_pct,
            "uncertainty_pct_upper": upper_pct,
            "lower_bound_kgco2e": lower_bound,
            "upper_bound_kgco2e": upper_bound,
            "method": "pedigree_analytical",
            "confidence_level_pct": Decimal("95.0"),
            "material_uncertainty_pct": material_unc,
        }

    # ==================================================================
    # PUBLIC API -- Health Check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the engine.

        Validates:

        1. Configuration is loaded.
        2. Emission factor table is populated.
        3. Unit conversion table is populated.
        4. Transport EF table is populated.
        5. A simple test calculation succeeds.

        Returns:
            Dictionary containing:
                - ``status``: "healthy" or "unhealthy".
                - ``agent_id``: Agent identifier.
                - ``version``: Agent version.
                - ``engine``: "average_data_calculator".
                - ``ef_count``: Number of emission factors loaded.
                - ``unit_count``: Number of unit conversions loaded.
                - ``transport_mode_count``: Number of transport modes.
                - ``material_category_count``: Number of category
                  defaults.
                - ``checks``: Detailed check results.
                - ``timestamp``: ISO 8601 UTC timestamp.

        Example:
            >>> health = engine.health_check()
            >>> health["status"]
            'healthy'
        """
        checks: Dict[str, bool] = {}
        status = "healthy"

        # Check 1: Configuration loaded
        try:
            _ = self._config.service_name
            checks["config_loaded"] = True
        except Exception:
            checks["config_loaded"] = False
            status = "unhealthy"

        # Check 2: EF table populated
        ef_count = len(PHYSICAL_EMISSION_FACTORS)
        checks["ef_table_populated"] = ef_count > 0
        if ef_count == 0:
            status = "unhealthy"

        # Check 3: Unit conversion table
        unit_count = len(UNIT_CONVERSION_TO_KG)
        checks["unit_table_populated"] = unit_count > 0
        if unit_count == 0:
            status = "unhealthy"

        # Check 4: Transport EF table
        transport_count = len(TRANSPORT_EFS)
        checks["transport_table_populated"] = transport_count > 0
        if transport_count == 0:
            status = "unhealthy"

        # Check 5: Test calculation
        try:
            test_item = ProcurementItem(
                description="Health check test item - steel coil",
                spend_amount=Decimal("1000"),
                quantity=Decimal("100"),
                quantity_unit="kg",
                material_category=MaterialCategory.RAW_METALS,
            )
            test_result = self.calculate_single(
                test_item,
                material_key="steel_world_avg",
            )
            expected_min = Decimal("130")
            expected_max = Decimal("140")
            calc_ok = (
                test_result.emissions_kgco2e >= expected_min
                and test_result.emissions_kgco2e <= expected_max
            )
            checks["test_calculation"] = calc_ok
            if not calc_ok:
                status = "unhealthy"
                logger.warning(
                    "Health check test calculation out of range: "
                    "%.4f (expected %.0f-%.0f)",
                    test_result.emissions_kgco2e,
                    expected_min,
                    expected_max,
                )
        except Exception as exc:
            checks["test_calculation"] = False
            status = "unhealthy"
            logger.error(
                "Health check test calculation failed: %s", exc
            )

        # Check 6: Category defaults complete
        cat_count = len(CATEGORY_DEFAULT_MATERIAL)
        checks["category_defaults_complete"] = (
            cat_count >= len(MaterialCategory)
        )

        return {
            "status": status,
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "average_data_calculator",
            "ef_count": ef_count,
            "unit_count": unit_count,
            "transport_mode_count": transport_count,
            "material_category_count": cat_count,
            "checks": checks,
            "timestamp": utcnow().isoformat(),
        }

    # ==================================================================
    # PUBLIC API -- Utility Methods
    # ==================================================================

    def list_supported_materials(self) -> List[str]:
        """Return sorted list of all supported material keys.

        Returns:
            Sorted list of material key strings.
        """
        return sorted(PHYSICAL_EMISSION_FACTORS.keys())

    def list_supported_units(self) -> List[str]:
        """Return sorted list of all supported unit strings.

        Returns:
            Sorted list of unit strings.
        """
        return sorted(UNIT_CONVERSION_TO_KG.keys())

    def list_supported_transport_modes(self) -> List[str]:
        """Return sorted list of all supported transport modes.

        Returns:
            Sorted list of transport mode strings.
        """
        return sorted(TRANSPORT_EFS.keys())

    def get_material_ef(self, material_key: str) -> Optional[Decimal]:
        """Get the emission factor for a material key.

        Convenience wrapper around ``resolve_ef_for_material`` without
        requiring an ``ef_source`` parameter.

        Args:
            material_key: The material key to look up.

        Returns:
            Emission factor in kgCO2e per kg, or None.
        """
        return PHYSICAL_EMISSION_FACTORS.get(material_key)

    def get_material_category(
        self, material_key: str,
    ) -> Optional[MaterialCategory]:
        """Get the MaterialCategory for a material key.

        Args:
            material_key: The material key to look up.

        Returns:
            MaterialCategory enum value, or None if not mapped.
        """
        return MATERIAL_CATEGORY_MAP.get(material_key)

    def get_material_uncertainty(
        self, material_key: str,
    ) -> Decimal:
        """Get the uncertainty percentage for a material key.

        Args:
            material_key: The material key to look up.

        Returns:
            Uncertainty percentage (e.g. Decimal("25")), or the
            default of 40% if not found.
        """
        return MATERIAL_UNCERTAINTY_PCT.get(
            material_key, Decimal("40")
        )

    def build_physical_ef_model(
        self, material_key: str,
    ) -> Optional[PhysicalEF]:
        """Build a PhysicalEF model object for a material key.

        Constructs a ``PhysicalEF`` Pydantic model from the embedded
        data tables for use in API responses and reporting.

        Args:
            material_key: Material key to build the model for.

        Returns:
            PhysicalEF model instance, or None if key not found.

        Example:
            >>> model = engine.build_physical_ef_model("steel_world_avg")
            >>> model.factor_kgco2e_per_kg
            Decimal('1.37')
        """
        ef = PHYSICAL_EMISSION_FACTORS.get(material_key)
        if ef is None:
            return None

        category = MATERIAL_CATEGORY_MAP.get(material_key)
        uncertainty = MATERIAL_UNCERTAINTY_PCT.get(material_key)

        # Determine source from material key
        source = self._infer_ef_source(material_key)

        return PhysicalEF(
            material_key=material_key,
            material_name=material_key.replace("_", " ").title(),
            factor_kgco2e_per_kg=ef,
            source=source,
            source_year=2023,
            region="GLOBAL",
            material_category=category,
            includes_transport=True,
            system_boundary="cradle_to_gate",
            uncertainty_pct=uncertainty,
        )

    # ==================================================================
    # PRIVATE -- Input Validation
    # ==================================================================

    def _validate_item_for_average_data(
        self, item: ProcurementItem,
    ) -> None:
        """Validate that a ProcurementItem has required fields.

        The average-data method requires:
        - ``quantity`` is not None and > 0.
        - ``quantity_unit`` is not None or empty.

        Args:
            item: The item to validate.

        Raises:
            ValueError: If required fields are missing.
        """
        if item.quantity is None:
            raise ValueError(
                f"Item '{item.item_id}': quantity is required "
                f"for average-data method"
            )
        if item.quantity <= ZERO:
            raise ValueError(
                f"Item '{item.item_id}': quantity must be positive, "
                f"got {item.quantity}"
            )
        if not item.quantity_unit:
            raise ValueError(
                f"Item '{item.item_id}': quantity_unit is required "
                f"for average-data method"
            )

    # ==================================================================
    # PRIVATE -- Material Resolution Helpers
    # ==================================================================

    def _resolve_from_description(
        self, description: str,
    ) -> Optional[str]:
        """Resolve material key from item description keywords.

        Performs case-insensitive longest-match against
        ``DESCRIPTION_KEYWORD_MAP``.  Longer keywords are checked
        first to ensure specificity (e.g. "recycled paper" beats
        "paper").

        Args:
            description: Item description text.

        Returns:
            Material key if matched, else None.
        """
        if not description:
            return None

        desc_lower = description.lower()
        # Sort by keyword length descending for longest match first
        sorted_keywords = sorted(
            DESCRIPTION_KEYWORD_MAP.keys(),
            key=len,
            reverse=True,
        )
        for keyword in sorted_keywords:
            if keyword in desc_lower:
                return DESCRIPTION_KEYWORD_MAP[keyword]
        return None

    def _get_material_density(
        self, material_key: Optional[str],
    ) -> Decimal:
        """Get material density in kg/L for volume conversion.

        Args:
            material_key: Material key for density lookup.

        Returns:
            Density in kg/L.  Defaults to 1.0 (water) if not found.
        """
        if material_key is None:
            return ONE
        return MATERIAL_DENSITY_KG_PER_L.get(material_key, ONE)

    def _get_piece_weight(
        self, material_key: Optional[str],
    ) -> Decimal:
        """Get per-piece weight in kg for count unit conversion.

        Args:
            material_key: Material key for piece weight lookup.

        Returns:
            Per-piece weight in kg.  Defaults to 1.0 if not found.
        """
        if material_key is None:
            return ONE
        return _DEFAULT_PIECE_WEIGHT_KG.get(material_key, ONE)

    def _infer_ef_source(
        self, material_key: str,
    ) -> PhysicalEFSource:
        """Infer the emission factor source from the material key.

        Maps material categories to their most likely data source.

        Args:
            material_key: The material key.

        Returns:
            PhysicalEFSource enum value.
        """
        category = MATERIAL_CATEGORY_MAP.get(material_key)
        if category is None:
            return PhysicalEFSource.DEFRA

        source_map: Dict[MaterialCategory, PhysicalEFSource] = {
            MaterialCategory.RAW_METALS: PhysicalEFSource.WORLD_STEEL,
            MaterialCategory.PLASTICS: PhysicalEFSource.PLASTICS_EUROPE,
            MaterialCategory.CONSTRUCTION: PhysicalEFSource.ICE,
            MaterialCategory.GLASS: PhysicalEFSource.ICE,
            MaterialCategory.WOOD: PhysicalEFSource.ICE,
            MaterialCategory.PAPER: PhysicalEFSource.CEPI,
            MaterialCategory.TEXTILES: PhysicalEFSource.DEFRA,
            MaterialCategory.ELECTRONICS: PhysicalEFSource.ECOINVENT,
            MaterialCategory.CHEMICALS: PhysicalEFSource.ECOINVENT,
            MaterialCategory.RUBBER: PhysicalEFSource.DEFRA,
        }
        return source_map.get(category, PhysicalEFSource.DEFRA)

    # ==================================================================
    # PRIVATE -- DQI Scoring Helpers
    # ==================================================================

    def _score_temporal_average_data(
        self,
        item: ProcurementItem,
        ef_source: PhysicalEFSource,
    ) -> Decimal:
        """Score temporal representativeness for average-data.

        Scoring rules:
        - Score 1: EF from reporting year (not typically available
          for average data).
        - Score 2: EF from within 3 years of reporting period.
          (Default for recognised databases updated recently.)
        - Score 3: EF from within 6 years.
        - Score 4: EF from within 10 years.
        - Score 5: EF older than 10 years.

        For average-data, we assume recognised databases (ecoinvent,
        ICE, World Steel, PlasticsEurope, CEPI) are within 3 years,
        yielding Score 2.  Custom sources get Score 4.

        Args:
            item: The procurement item (for period context).
            ef_source: The EF source database.

        Returns:
            Temporal DQI score (1-5 Decimal).
        """
        recognised_recent: frozenset = frozenset({
            PhysicalEFSource.ECOINVENT,
            PhysicalEFSource.GABI,
            PhysicalEFSource.WORLD_STEEL,
            PhysicalEFSource.PLASTICS_EUROPE,
            PhysicalEFSource.IAI,
            PhysicalEFSource.CEPI,
            PhysicalEFSource.ICE,
        })
        if ef_source in recognised_recent:
            return Decimal("2.0")
        if ef_source == PhysicalEFSource.DEFRA:
            return Decimal("2.0")
        # Custom or unknown
        return Decimal("4.0")

    def _score_geographical_average_data(
        self,
        item: ProcurementItem,
        ef_source: PhysicalEFSource,
    ) -> Decimal:
        """Score geographical representativeness for average-data.

        Average-data EFs are typically global or regional averages.
        Country-specific ecoinvent factors get Score 2.  Global
        averages from World Steel, IAI, etc. get Score 3.  DEFRA
        (UK-specific) gets Score 2 for UK items, Score 3 otherwise.
        Custom gets Score 4.

        Args:
            item: The procurement item.
            ef_source: The EF source database.

        Returns:
            Geographical DQI score (1-5 Decimal).
        """
        country_specific = frozenset({
            PhysicalEFSource.ECOINVENT,
            PhysicalEFSource.GABI,
        })
        if ef_source in country_specific:
            return Decimal("2.0")
        if ef_source == PhysicalEFSource.DEFRA:
            return Decimal("2.5")
        if ef_source == PhysicalEFSource.CUSTOM:
            return Decimal("4.0")
        # Global/regional averages
        return Decimal("3.0")

    def _score_technological_average_data(
        self,
        item: ProcurementItem,
        result: AverageDataResult,
    ) -> Decimal:
        """Score technological representativeness for average-data.

        Industry-average EFs may not match the specific production
        technology.  Items with more specific material keys (e.g.
        ``steel_primary_bof`` vs ``steel_world_avg``) get better
        scores.

        Args:
            item: The procurement item.
            result: The calculation result.

        Returns:
            Technological DQI score (1-5 Decimal).
        """
        key = result.material_key
        # Specific process variants get better score
        specific_indicators = [
            "_primary_", "_secondary_", "_bof", "_eaf",
            "_virgin_", "_recycled", "_organic",
            "_cem_i", "_30mpa", "_50mpa",
            "_softwood_", "_hardwood_", "_glulam",
        ]
        for indicator in specific_indicators:
            if indicator in key:
                return Decimal("2.0")

        # Generic/average keys
        if "_avg" in key or "_general" in key or "world" in key:
            return Decimal("3.5")

        return Decimal("3.0")

    def _score_completeness_average_data(
        self,
        result: AverageDataResult,
    ) -> Decimal:
        """Score completeness for average-data.

        Checks whether transport and waste/loss are included.

        Args:
            result: The calculation result.

        Returns:
            Completeness DQI score (1-5 Decimal).
        """
        has_transport = result.transport_emissions_kgco2e > ZERO
        has_waste = result.waste_loss_factor > ONE

        if has_transport and has_waste:
            return Decimal("2.0")
        if has_transport or has_waste:
            return Decimal("2.5")
        return Decimal("3.0")

    def _score_reliability_average_data(
        self,
        ef_source: PhysicalEFSource,
    ) -> Decimal:
        """Score reliability for average-data.

        Recognised databases with peer-reviewed data get Score 2.
        Government databases (DEFRA) get Score 2.  Custom/unknown
        sources get Score 4.

        Args:
            ef_source: The EF source database.

        Returns:
            Reliability DQI score (1-5 Decimal).
        """
        high_reliability = frozenset({
            PhysicalEFSource.ECOINVENT,
            PhysicalEFSource.GABI,
        })
        medium_reliability = frozenset({
            PhysicalEFSource.WORLD_STEEL,
            PhysicalEFSource.IAI,
            PhysicalEFSource.PLASTICS_EUROPE,
            PhysicalEFSource.ICE,
            PhysicalEFSource.CEPI,
            PhysicalEFSource.DEFRA,
        })
        if ef_source in high_reliability:
            return Decimal("1.5")
        if ef_source in medium_reliability:
            return Decimal("2.0")
        return Decimal("4.0")

    def _determine_quality_tier(
        self, composite_score: Decimal,
    ) -> str:
        """Determine quality tier label from composite DQI score.

        Uses the DQI_QUALITY_TIERS ranges from models.py.

        Args:
            composite_score: The composite DQI score.

        Returns:
            Quality tier label string.
        """
        from greenlang.agents.mrv.purchased_goods_services.models import (
            DQI_QUALITY_TIERS,
        )
        for tier_name, (min_val, max_val) in DQI_QUALITY_TIERS.items():
            if min_val <= composite_score < max_val:
                return tier_name
        return "Very Poor"

    def _determine_ef_hierarchy_level(
        self, ef_source: PhysicalEFSource,
    ) -> int:
        """Determine EF hierarchy level for average-data sources.

        Per GHG Protocol Scope 3 Technical Guidance Section 1.4:
        - Level 3: Product LCA from ecoinvent/GaBi
        - Level 4: Material average from ICE/DEFRA
        - Level 5: Industry average physical EF

        Args:
            ef_source: The EF source database.

        Returns:
            EF hierarchy level (1-8, lower is better).
        """
        level_map: Dict[PhysicalEFSource, int] = {
            PhysicalEFSource.ECOINVENT: 3,
            PhysicalEFSource.GABI: 3,
            PhysicalEFSource.ICE: 4,
            PhysicalEFSource.DEFRA: 4,
            PhysicalEFSource.WORLD_STEEL: 4,
            PhysicalEFSource.PLASTICS_EUROPE: 4,
            PhysicalEFSource.IAI: 4,
            PhysicalEFSource.CEPI: 4,
            PhysicalEFSource.CUSTOM: 5,
        }
        return level_map.get(ef_source, 5)

    def _compute_pedigree_uncertainty(
        self, composite_score: Decimal,
    ) -> Decimal:
        """Compute pedigree uncertainty factor from composite DQI.

        Maps the composite score to the closest DQIScore label,
        then looks up the pedigree uncertainty factor.

        Args:
            composite_score: The composite DQI score.

        Returns:
            Pedigree uncertainty factor (>= 1.0).
        """
        if composite_score <= Decimal("1.5"):
            score_label = DQIScore.VERY_GOOD
        elif composite_score <= Decimal("2.5"):
            score_label = DQIScore.GOOD
        elif composite_score <= Decimal("3.5"):
            score_label = DQIScore.FAIR
        elif composite_score <= Decimal("4.5"):
            score_label = DQIScore.POOR
        else:
            score_label = DQIScore.VERY_POOR

        return PEDIGREE_UNCERTAINTY_FACTORS.get(
            score_label, Decimal("1.50")
        )

    # ==================================================================
    # PRIVATE -- Provenance
    # ==================================================================

    def _compute_result_provenance(
        self,
        item_id: str,
        material_key: str,
        quantity_kg: Decimal,
        ef_kgco2e: Decimal,
        waste_loss_factor: Decimal,
        base_emissions: Decimal,
        transport_emissions: Decimal,
        total_emissions: Decimal,
    ) -> str:
        """Compute SHA-256 provenance hash for a calculation result.

        Hashes all inputs and outputs into a single deterministic
        digest for the audit trail.

        Args:
            item_id: Procurement item identifier.
            material_key: Material key used.
            quantity_kg: Quantity in kilograms.
            ef_kgco2e: Emission factor applied.
            waste_loss_factor: Waste/loss factor applied.
            base_emissions: Base emissions (no transport).
            transport_emissions: Transport adder emissions.
            total_emissions: Total emissions (base + transport).

        Returns:
            64-character hex SHA-256 digest.
        """
        data = {
            "engine": "average_data_calculator",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "item_id": item_id,
            "material_key": material_key,
            "quantity_kg": str(quantity_kg),
            "ef_kgco2e_per_kg": str(ef_kgco2e),
            "waste_loss_factor": str(waste_loss_factor),
            "base_emissions_kgco2e": str(base_emissions),
            "transport_emissions_kgco2e": str(transport_emissions),
            "total_emissions_kgco2e": str(total_emissions),
            "calculation_method": CalculationMethod.AVERAGE_DATA.value,
            "timestamp": utcnow().isoformat(),
        }
        return _sha256(data)

    # ==================================================================
    # PRIVATE -- Metrics
    # ==================================================================

    def _record_calculation_metric(
        self,
        item: ProcurementItem,
        result: AverageDataResult,
        duration_s: float,
        status: str,
    ) -> None:
        """Record Prometheus metrics for a calculation.

        Args:
            item: The procurement item.
            result: The calculation result.
            duration_s: Duration in seconds.
            status: "success" or "failed".
        """
        try:
            category = (
                MATERIAL_CATEGORY_MAP.get(result.material_key, "other")
            )
            if isinstance(category, MaterialCategory):
                category = category.value

            self._metrics.record_calculation(
                tenant_id=self._config.default_tenant,
                method=CalculationMethod.AVERAGE_DATA.value,
                status=status,
                duration_s=duration_s,
                emissions_kgco2e=float(
                    result.total_with_transport_kgco2e
                ),
                material_category=category,
            )
        except Exception as exc:
            logger.debug(
                "Failed to record metric: %s", exc
            )

    # ==================================================================
    # PRIVATE -- Math Utilities
    # ==================================================================

    def _decimal_sqrt(self, value: Decimal) -> Decimal:
        """Compute square root of a Decimal using Newton's method.

        Iterates until convergence within the configured decimal
        places, ensuring deterministic Decimal-only arithmetic.

        Args:
            value: Non-negative Decimal value.

        Returns:
            Square root as Decimal, quantized.

        Raises:
            ValueError: If value is negative.
        """
        if value < ZERO:
            raise ValueError(
                f"Cannot compute sqrt of negative value: {value}"
            )
        if value == ZERO:
            return _q(ZERO, self._decimal_places)

        # Initial guess
        guess = value / Decimal("2")
        two = Decimal("2")
        tolerance = Decimal(10) ** -(self._decimal_places + 2)

        for _ in range(100):  # Max iterations
            next_guess = (guess + value / guess) / two
            diff = abs(next_guess - guess)
            if diff < tolerance:
                return _q(next_guess, self._decimal_places)
            guess = next_guess

        return _q(guess, self._decimal_places)

    # ==================================================================
    # PRIVATE -- Batch DQI Scoring (convenience)
    # ==================================================================

    def score_dqi_batch(
        self,
        items_and_results: List[
            Tuple[ProcurementItem, PhysicalEFSource, AverageDataResult]
        ],
    ) -> List[DQIAssessment]:
        """Score DQI for a batch of calculation results.

        Args:
            items_and_results: List of (item, ef_source, result) tuples.

        Returns:
            List of DQIAssessment objects.
        """
        assessments: List[DQIAssessment] = []
        for item, source, result in items_and_results:
            try:
                dqi = self.score_dqi_average_data(item, source, result)
                assessments.append(dqi)
            except Exception as exc:
                logger.warning(
                    "DQI scoring failed for %s: %s",
                    item.item_id,
                    exc,
                )
        return assessments

    def compute_weighted_dqi(
        self,
        results: List[AverageDataResult],
        assessments: List[DQIAssessment],
    ) -> Decimal:
        """Compute emission-weighted composite DQI score.

        Weights each item's composite DQI score by its emission
        share of the total, producing a single portfolio-level
        quality indicator.

        Args:
            results: List of AverageDataResult objects.
            assessments: Corresponding list of DQIAssessment objects.

        Returns:
            Emission-weighted composite DQI score (1-5 Decimal).
        """
        if not results or not assessments:
            return Decimal("5.0")

        # Build lookup
        dqi_map: Dict[str, Decimal] = {
            a.item_id: a.composite_score for a in assessments
        }

        total_emissions = _q(ZERO, self._decimal_places)
        weighted_sum = _q(ZERO, self._decimal_places)

        for r in results:
            emissions = r.total_with_transport_kgco2e
            composite = dqi_map.get(r.item_id, Decimal("5.0"))
            weighted_sum += emissions * composite
            total_emissions += emissions

        if total_emissions == ZERO:
            return Decimal("5.0")

        weighted_dqi = _q(
            weighted_sum / total_emissions,
            self._decimal_places,
        )

        # Clamp to 1-5 range
        if weighted_dqi < ONE:
            weighted_dqi = ONE
        if weighted_dqi > Decimal("5.0"):
            weighted_dqi = Decimal("5.0")

        return weighted_dqi

    # ==================================================================
    # PUBLIC API -- Summary Statistics
    # ==================================================================

    def compute_summary_statistics(
        self,
        results: List[AverageDataResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics for a list of results.

        Args:
            results: List of AverageDataResult objects.

        Returns:
            Dictionary with summary statistics:
                - ``item_count``: Number of results.
                - ``total_emissions_kgco2e``: Sum of base emissions.
                - ``total_transport_kgco2e``: Sum of transport.
                - ``total_with_transport_kgco2e``: Grand total.
                - ``total_emissions_tco2e``: In tonnes.
                - ``total_quantity_kg``: Total quantity.
                - ``avg_ef_kgco2e_per_kg``: Mean emission factor.
                - ``max_emissions_kgco2e``: Highest single-item.
                - ``min_emissions_kgco2e``: Lowest single-item.
                - ``unique_materials``: Count of unique material keys.
                - ``material_list``: Sorted unique material keys.
        """
        if not results:
            return {
                "item_count": 0,
                "total_emissions_kgco2e": _q(ZERO, self._decimal_places),
                "total_transport_kgco2e": _q(ZERO, self._decimal_places),
                "total_with_transport_kgco2e": _q(
                    ZERO, self._decimal_places
                ),
                "total_emissions_tco2e": _q(ZERO, self._decimal_places),
                "total_quantity_kg": _q(ZERO, self._decimal_places),
                "avg_ef_kgco2e_per_kg": _q(ZERO, self._decimal_places),
                "max_emissions_kgco2e": _q(ZERO, self._decimal_places),
                "min_emissions_kgco2e": _q(ZERO, self._decimal_places),
                "unique_materials": 0,
                "material_list": [],
            }

        total_base = _q(ZERO, self._decimal_places)
        total_transport = _q(ZERO, self._decimal_places)
        total_combined = _q(ZERO, self._decimal_places)
        total_tco2e = _q(ZERO, self._decimal_places)
        total_qty = _q(ZERO, self._decimal_places)
        ef_sum = _q(ZERO, self._decimal_places)
        max_em = _q(ZERO, self._decimal_places)
        min_em: Optional[Decimal] = None
        materials: set = set()

        for r in results:
            total_base += r.emissions_kgco2e
            total_transport += r.transport_emissions_kgco2e
            total_combined += r.total_with_transport_kgco2e
            total_tco2e += r.emissions_tco2e
            total_qty += r.quantity_kg
            ef_sum += r.ef_kgco2e_per_kg
            if r.total_with_transport_kgco2e > max_em:
                max_em = r.total_with_transport_kgco2e
            if min_em is None or r.total_with_transport_kgco2e < min_em:
                min_em = r.total_with_transport_kgco2e
            materials.add(r.material_key)

        count = Decimal(str(len(results)))
        avg_ef = _q(ef_sum / count, self._decimal_places) if count > ZERO else ZERO

        return {
            "item_count": len(results),
            "total_emissions_kgco2e": _q(total_base, self._decimal_places),
            "total_transport_kgco2e": _q(
                total_transport, self._decimal_places
            ),
            "total_with_transport_kgco2e": _q(
                total_combined, self._decimal_places
            ),
            "total_emissions_tco2e": _q(total_tco2e, self._decimal_places),
            "total_quantity_kg": _q(total_qty, self._decimal_places),
            "avg_ef_kgco2e_per_kg": avg_ef,
            "max_emissions_kgco2e": _q(max_em, self._decimal_places),
            "min_emissions_kgco2e": _q(
                min_em if min_em is not None else ZERO,
                self._decimal_places,
            ),
            "unique_materials": len(materials),
            "material_list": sorted(materials),
        }

    # ==================================================================
    # PUBLIC API -- Emission Factor Enrichment
    # ==================================================================

    def enrich_physical_record(
        self,
        item: ProcurementItem,
        material_key: Optional[str] = None,
        ef_source: PhysicalEFSource = PhysicalEFSource.DEFRA,
        includes_transport: bool = True,
        transport_distance_km: Decimal = ZERO,
        transport_mode: Optional[str] = None,
        waste_loss_factor: Decimal = ONE,
    ) -> PhysicalRecord:
        """Create an enriched PhysicalRecord from a ProcurementItem.

        Resolves the material key, converts the quantity to kg, and
        populates all PhysicalRecord fields.  This is useful for
        pipeline stage separation where enrichment and calculation
        are distinct steps.

        Args:
            item: The procurement item to enrich.
            material_key: Explicit material key (resolved if None).
            ef_source: EF source database.
            includes_transport: Whether EF includes transport.
            transport_distance_km: Transport distance in km.
            transport_mode: Transport mode string.
            waste_loss_factor: Waste/loss factor (>= 1.0).

        Returns:
            Enriched PhysicalRecord ready for calculation.

        Raises:
            ValueError: If material key cannot be resolved or item
                lacks quantity data.
        """
        self._validate_item_for_average_data(item)

        resolved_key = material_key
        if resolved_key is None:
            resolved_key = self.resolve_material_key(item)
        if resolved_key is None:
            raise ValueError(
                f"Cannot resolve material key for item "
                f"'{item.item_id}'"
            )

        quantity_kg = self.convert_to_kg(
            item.quantity,  # type: ignore[arg-type]
            item.quantity_unit,  # type: ignore[arg-type]
            resolved_key,
        )

        return PhysicalRecord(
            item=item,
            quantity_kg=quantity_kg,
            material_key=resolved_key,
            ef_source=ef_source,
            includes_transport=includes_transport,
            transport_distance_km=transport_distance_km,
            transport_mode=transport_mode,
            waste_loss_factor=max(waste_loss_factor, ONE),
        )

    # ==================================================================
    # PUBLIC API -- Ranking and Hotspot Identification
    # ==================================================================

    def rank_by_emissions(
        self,
        results: List[AverageDataResult],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Rank results by emissions (highest first) for hot-spot analysis.

        Args:
            results: List of AverageDataResult objects.
            top_n: Number of top items to return (default 20).

        Returns:
            List of dicts with rank, item_id, material_key, emissions,
            and cumulative percentage.
        """
        if not results:
            return []

        # Sort descending by total emissions
        sorted_results = sorted(
            results,
            key=lambda r: r.total_with_transport_kgco2e,
            reverse=True,
        )

        total = _q(ZERO, self._decimal_places)
        for r in sorted_results:
            total += r.total_with_transport_kgco2e

        if total == ZERO:
            return []

        ranked: List[Dict[str, Any]] = []
        cumulative = _q(ZERO, self._decimal_places)
        for idx, r in enumerate(sorted_results[:top_n]):
            pct = _q(
                (r.total_with_transport_kgco2e / total) * ONE_HUNDRED,
                self._decimal_places,
            )
            cumulative = _q(cumulative + pct, self._decimal_places)
            ranked.append({
                "rank": idx + 1,
                "item_id": r.item_id,
                "material_key": r.material_key,
                "emissions_kgco2e": r.total_with_transport_kgco2e,
                "emissions_tco2e": r.emissions_tco2e,
                "emissions_pct": pct,
                "cumulative_pct": cumulative,
            })

        return ranked

    # ==================================================================
    # PUBLIC API -- Emission Factor Table Export
    # ==================================================================

    def export_ef_table(self) -> List[Dict[str, Any]]:
        """Export the complete physical emission factor table.

        Returns a list of dictionaries representing all embedded
        emission factors, suitable for API response or CSV export.

        Returns:
            List of dicts with material_key, factor, category,
            source, and uncertainty.
        """
        table: List[Dict[str, Any]] = []
        for key in sorted(PHYSICAL_EMISSION_FACTORS.keys()):
            ef = PHYSICAL_EMISSION_FACTORS[key]
            category = MATERIAL_CATEGORY_MAP.get(key)
            source = self._infer_ef_source(key)
            uncertainty = MATERIAL_UNCERTAINTY_PCT.get(key)
            table.append({
                "material_key": key,
                "factor_kgco2e_per_kg": str(ef),
                "material_category": (
                    category.value if category else None
                ),
                "ef_source": source.value,
                "uncertainty_pct": (
                    str(uncertainty) if uncertainty else None
                ),
                "system_boundary": "cradle_to_gate",
            })
        return table

    # ==================================================================
    # PUBLIC API -- Transport Mode Comparison
    # ==================================================================

    def compare_transport_modes(
        self,
        quantity_kg: Decimal,
        distance_km: Decimal,
    ) -> Dict[str, Decimal]:
        """Compare transport emissions across all modes.

        Calculates transport emissions for the same quantity and
        distance across all supported transport modes, enabling
        mode selection optimisation.

        Args:
            quantity_kg: Quantity of goods in kilograms.
            distance_km: Transport distance in kilometres.

        Returns:
            Dictionary keyed by transport mode, valued with
            emissions in kgCO2e.
        """
        comparison: Dict[str, Decimal] = {}
        for mode in sorted(TRANSPORT_EFS.keys()):
            try:
                emissions = self.calculate_transport_emissions(
                    quantity_kg, distance_km, mode
                )
                comparison[mode] = emissions
            except ValueError:
                continue
        return comparison

    # ==================================================================
    # PUBLIC API -- Validation
    # ==================================================================

    def validate_material_key(self, material_key: str) -> bool:
        """Check whether a material key exists in the EF table.

        Args:
            material_key: Material key to validate.

        Returns:
            True if the key has a corresponding emission factor.
        """
        return material_key in PHYSICAL_EMISSION_FACTORS

    def validate_unit(self, unit: str) -> bool:
        """Check whether a unit string is supported.

        Args:
            unit: Unit string to validate.

        Returns:
            True if the unit is in the conversion table.
        """
        if not unit:
            return False
        normalised = unit.strip().lower().replace(" ", "_")
        return normalised in UNIT_CONVERSION_TO_KG

    def validate_transport_mode(self, mode: str) -> bool:
        """Check whether a transport mode is supported.

        Args:
            mode: Transport mode string to validate.

        Returns:
            True if the mode is in the transport EF table.
        """
        if not mode:
            return False
        normalised = mode.strip().lower().replace(" ", "_")
        return normalised in TRANSPORT_EFS
