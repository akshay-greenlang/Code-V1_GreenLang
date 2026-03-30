# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine -- AGENT-MRV-023 Engine 3 of 7

This module implements the AverageDataCalculatorEngine for the Processing of
Sold Products Agent (GL-MRV-S3-010).  The engine calculates GHG emissions
using the *average-data method* from the GHG Protocol Scope 3 Technical
Guidance, Chapter 6 (Category 10), where emissions are determined by
multiplying quantities of intermediate products sold by industry-average
processing emission factors.

Core Formulae::

    Formula D -- Process-Specific EF:
        E_cat10 = Sum_i( Q_sold_i x EF_process_i )

    Formula E -- Energy Intensity:
        E_cat10 = Sum_i( Q_sold_i x EI_i x EF_grid_avg )

    Formula F (partial) -- Sector Benchmark:
        E_cat10 = Sum_i( Q_sold_i x EF_sector_i )

The engine supports:

- **Process-specific EF calculation** via ``calculate_process_ef`` using
  12 product category emission factors from industry databases.
- **Energy intensity calculation** via ``calculate_energy_intensity`` using
  18 processing type energy intensities combined with grid emission factors.
- **Sector benchmark calculation** via ``calculate_sector_benchmark`` using
  sector-level processing emission benchmarks.
- **Multi-step chain calculation** via ``calculate_chain_emissions`` for
  8 standard multi-step processing chains with combined EFs.
- **Single product breakdown** via ``calculate_product_emissions``.
- **Dispatcher** via ``calculate`` that selects the appropriate sub-method.
- **5-dimension DQI scoring** per GHG Protocol Scope 3 Standard
  Chapter 7 with medium-quality default scores for average-data methods.
- **Uncertainty estimation** per method type with configurable ranges.
- **SHA-256 provenance tracking** for complete audit trails.

Product Categories (12):
    METALS_FERROUS, METALS_NON_FERROUS, PLASTICS_THERMOPLASTIC,
    PLASTICS_THERMOSET, CHEMICALS, FOOD_INGREDIENTS, TEXTILES,
    ELECTRONICS, GLASS_CERAMICS, WOOD_PAPER, MINERALS, AGRICULTURAL

Processing Types (18):
    MACHINING, STAMPING, WELDING, HEAT_TREATMENT, INJECTION_MOLDING,
    EXTRUSION, BLOW_MOLDING, CASTING, FORGING, COATING, ASSEMBLY,
    CHEMICAL_REACTION, REFINING, MILLING, DRYING, SINTERING,
    FERMENTATION, TEXTILE_FINISHING

Multi-Step Chains (8):
    steel_automotive_parts, aluminum_beverage_cans, plastic_packaging,
    semiconductor_chips, food_products, textile_garments, glass_bottles,
    paper_products

Thread Safety:
    The engine is implemented as a thread-safe singleton using
    ``threading.RLock`` with double-checked locking.  All internal
    state is either immutable (constant tables) or protected by the
    instance lock where necessary.

Zero-Hallucination Guarantee:
    All numeric operations use ``Decimal`` with explicit quantization.
    No LLM calls are made for any numeric calculation.  Emission
    factors are sourced exclusively from the deterministic constant
    tables embedded in this module, derived from PRD-AGENT-MRV-023
    Section 5.1, 5.2, and 5.6.

Emission Factor Sources:
    - DEFRA 2024 emission conversion factors
    - EPA GHG Emission Factors Hub 2024
    - ecoinvent v3.11 LCA database
    - Industry associations (World Steel, IAI, PlasticsEurope)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-023 Processing of Sold Products (GL-MRV-S3-010)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AverageDataCalculatorEngine",
    "PROCESSING_EMISSION_FACTORS",
    "ENERGY_INTENSITY_FACTORS",
    "GRID_EMISSION_FACTORS",
    "PROCESSING_CHAINS",
    "SECTOR_BENCHMARK_FACTORS",
    "ProductCategory",
    "ProcessingType",
    "AverageDataMethod",
    "CalculationResult",
    "ProductBreakdown",
    "DataQualityScore",
    "UncertaintyResult",
]

# ---------------------------------------------------------------------------
# Agent-level constants
# ---------------------------------------------------------------------------

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

# ---------------------------------------------------------------------------
# Decimal constants
# ---------------------------------------------------------------------------

DECIMAL_PLACES: int = 8
ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
ONE_HUNDRED: Decimal = Decimal("100")
ONE_THOUSAND: Decimal = Decimal("1000")
_PRECISION: Decimal = Decimal(10) ** -DECIMAL_PLACES

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProductCategory(str, Enum):
    """Intermediate product categories for Category 10 processing.

    These 12 categories cover the range of intermediate products that
    a reporting company may sell to downstream processors. Each category
    has an associated default processing emission factor from industry
    databases (DEFRA, EPA, ecoinvent).
    """

    METALS_FERROUS = "METALS_FERROUS"
    METALS_NON_FERROUS = "METALS_NON_FERROUS"
    PLASTICS_THERMOPLASTIC = "PLASTICS_THERMOPLASTIC"
    PLASTICS_THERMOSET = "PLASTICS_THERMOSET"
    CHEMICALS = "CHEMICALS"
    FOOD_INGREDIENTS = "FOOD_INGREDIENTS"
    TEXTILES = "TEXTILES"
    ELECTRONICS = "ELECTRONICS"
    GLASS_CERAMICS = "GLASS_CERAMICS"
    WOOD_PAPER = "WOOD_PAPER"
    MINERALS = "MINERALS"
    AGRICULTURAL = "AGRICULTURAL"

class ProcessingType(str, Enum):
    """Processing types for downstream manufacturing operations.

    These 18 processing types represent the standard manufacturing
    operations that downstream processors apply to intermediate products.
    Each type has an associated energy intensity (kWh per tonne) derived
    from industry benchmarks and LCA databases.

from greenlang.schemas import utcnow
    """

    MACHINING = "MACHINING"
    STAMPING = "STAMPING"
    WELDING = "WELDING"
    HEAT_TREATMENT = "HEAT_TREATMENT"
    INJECTION_MOLDING = "INJECTION_MOLDING"
    EXTRUSION = "EXTRUSION"
    BLOW_MOLDING = "BLOW_MOLDING"
    CASTING = "CASTING"
    FORGING = "FORGING"
    COATING = "COATING"
    ASSEMBLY = "ASSEMBLY"
    CHEMICAL_REACTION = "CHEMICAL_REACTION"
    REFINING = "REFINING"
    MILLING = "MILLING"
    DRYING = "DRYING"
    SINTERING = "SINTERING"
    FERMENTATION = "FERMENTATION"
    TEXTILE_FINISHING = "TEXTILE_FINISHING"

class AverageDataMethod(str, Enum):
    """Sub-methods available for average-data calculation.

    PROCESS_EF: Uses product category emission factors (Formula D).
    ENERGY_INTENSITY: Uses energy intensity x grid EF (Formula E).
    SECTOR_BENCHMARK: Uses sector benchmark EFs.
    CHAIN: Uses multi-step processing chain EFs.
    """

    PROCESS_EF = "process_ef"
    ENERGY_INTENSITY = "energy_intensity"
    SECTOR_BENCHMARK = "sector_benchmark"
    CHAIN = "chain"

class AllocationMethod(str, Enum):
    """Allocation methods for multi-use intermediate products.

    MASS: Allocate by mass fraction of end uses.
    REVENUE: Allocate by revenue fraction of end uses.
    UNITS: Allocate by unit count fraction of end uses.
    EQUAL: Allocate equally across end uses.
    """

    MASS = "mass"
    REVENUE = "revenue"
    UNITS = "units"
    EQUAL = "equal"

# ---------------------------------------------------------------------------
# Emission Factor Tables -- All Decimal for Deterministic Arithmetic
# ---------------------------------------------------------------------------

# PRD Section 5.1: Processing Emission Factors by Product Category (kgCO2e/tonne)
PROCESSING_EMISSION_FACTORS: Dict[str, Decimal] = {
    ProductCategory.METALS_FERROUS.value: Decimal("280"),
    ProductCategory.METALS_NON_FERROUS.value: Decimal("380"),
    ProductCategory.PLASTICS_THERMOPLASTIC.value: Decimal("520"),
    ProductCategory.PLASTICS_THERMOSET.value: Decimal("450"),
    ProductCategory.CHEMICALS.value: Decimal("680"),
    ProductCategory.FOOD_INGREDIENTS.value: Decimal("130"),
    ProductCategory.TEXTILES.value: Decimal("350"),
    ProductCategory.ELECTRONICS.value: Decimal("950"),
    ProductCategory.GLASS_CERAMICS.value: Decimal("580"),
    ProductCategory.WOOD_PAPER.value: Decimal("190"),
    ProductCategory.MINERALS.value: Decimal("250"),
    ProductCategory.AGRICULTURAL.value: Decimal("110"),
}

# PRD Section 5.1: Uncertainty percentages by product category
CATEGORY_UNCERTAINTY_PCT: Dict[str, Decimal] = {
    ProductCategory.METALS_FERROUS.value: Decimal("25"),
    ProductCategory.METALS_NON_FERROUS.value: Decimal("25"),
    ProductCategory.PLASTICS_THERMOPLASTIC.value: Decimal("30"),
    ProductCategory.PLASTICS_THERMOSET.value: Decimal("30"),
    ProductCategory.CHEMICALS.value: Decimal("35"),
    ProductCategory.FOOD_INGREDIENTS.value: Decimal("20"),
    ProductCategory.TEXTILES.value: Decimal("30"),
    ProductCategory.ELECTRONICS.value: Decimal("35"),
    ProductCategory.GLASS_CERAMICS.value: Decimal("25"),
    ProductCategory.WOOD_PAPER.value: Decimal("20"),
    ProductCategory.MINERALS.value: Decimal("25"),
    ProductCategory.AGRICULTURAL.value: Decimal("20"),
}

# PRD Section 5.2: Energy Intensity by Processing Type (kWh/tonne, default/mid value)
ENERGY_INTENSITY_FACTORS: Dict[str, Decimal] = {
    ProcessingType.MACHINING.value: Decimal("280"),
    ProcessingType.STAMPING.value: Decimal("140"),
    ProcessingType.WELDING.value: Decimal("220"),
    ProcessingType.HEAT_TREATMENT.value: Decimal("380"),
    ProcessingType.INJECTION_MOLDING.value: Decimal("520"),
    ProcessingType.EXTRUSION.value: Decimal("340"),
    ProcessingType.BLOW_MOLDING.value: Decimal("400"),
    ProcessingType.CASTING.value: Decimal("750"),
    ProcessingType.FORGING.value: Decimal("580"),
    ProcessingType.COATING.value: Decimal("120"),
    ProcessingType.ASSEMBLY.value: Decimal("45"),
    ProcessingType.CHEMICAL_REACTION.value: Decimal("1100"),
    ProcessingType.REFINING.value: Decimal("900"),
    ProcessingType.MILLING.value: Decimal("190"),
    ProcessingType.DRYING.value: Decimal("310"),
    ProcessingType.SINTERING.value: Decimal("1200"),
    ProcessingType.FERMENTATION.value: Decimal("160"),
    ProcessingType.TEXTILE_FINISHING.value: Decimal("420"),
}

# PRD Section 5.2: Energy Intensity ranges (low, mid, high) kWh/tonne
ENERGY_INTENSITY_RANGES: Dict[str, Tuple[Decimal, Decimal, Decimal]] = {
    ProcessingType.MACHINING.value: (Decimal("150"), Decimal("280"), Decimal("450")),
    ProcessingType.STAMPING.value: (Decimal("80"), Decimal("140"), Decimal("200")),
    ProcessingType.WELDING.value: (Decimal("100"), Decimal("220"), Decimal("350")),
    ProcessingType.HEAT_TREATMENT.value: (Decimal("200"), Decimal("380"), Decimal("600")),
    ProcessingType.INJECTION_MOLDING.value: (Decimal("300"), Decimal("520"), Decimal("800")),
    ProcessingType.EXTRUSION.value: (Decimal("200"), Decimal("340"), Decimal("500")),
    ProcessingType.BLOW_MOLDING.value: (Decimal("250"), Decimal("400"), Decimal("600")),
    ProcessingType.CASTING.value: (Decimal("400"), Decimal("750"), Decimal("1200")),
    ProcessingType.FORGING.value: (Decimal("300"), Decimal("580"), Decimal("900")),
    ProcessingType.COATING.value: (Decimal("50"), Decimal("120"), Decimal("200")),
    ProcessingType.ASSEMBLY.value: (Decimal("20"), Decimal("45"), Decimal("80")),
    ProcessingType.CHEMICAL_REACTION.value: (Decimal("500"), Decimal("1100"), Decimal("2000")),
    ProcessingType.REFINING.value: (Decimal("400"), Decimal("900"), Decimal("1500")),
    ProcessingType.MILLING.value: (Decimal("100"), Decimal("190"), Decimal("300")),
    ProcessingType.DRYING.value: (Decimal("150"), Decimal("310"), Decimal("500")),
    ProcessingType.SINTERING.value: (Decimal("600"), Decimal("1200"), Decimal("2000")),
    ProcessingType.FERMENTATION.value: (Decimal("80"), Decimal("160"), Decimal("250")),
    ProcessingType.TEXTILE_FINISHING.value: (Decimal("200"), Decimal("420"), Decimal("700")),
}

# PRD Section 5.4: Grid Emission Factors (kgCO2e/kWh by Country/Region)
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("0.417"),
    "GB": Decimal("0.233"),
    "DE": Decimal("0.348"),
    "FR": Decimal("0.052"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "JP": Decimal("0.462"),
    "KR": Decimal("0.424"),
    "BR": Decimal("0.075"),
    "CA": Decimal("0.120"),
    "AU": Decimal("0.656"),
    "MX": Decimal("0.431"),
    "IT": Decimal("0.256"),
    "ES": Decimal("0.175"),
    "PL": Decimal("0.635"),
    "GLOBAL": Decimal("0.475"),
}

# PRD Section 5.6: Multi-Step Processing Chains (combined EFs in kgCO2e/tonne)
PROCESSING_CHAINS: Dict[str, Dict[str, Any]] = {
    "steel_automotive_parts": {
        "description": "Stamping -> Welding -> Coating",
        "steps": [ProcessingType.STAMPING.value, ProcessingType.WELDING.value, ProcessingType.COATING.value],
        "combined_ef": Decimal("480"),
        "applicable_categories": [ProductCategory.METALS_FERROUS.value],
    },
    "aluminum_beverage_cans": {
        "description": "Stamping -> Coating -> Assembly",
        "steps": [ProcessingType.STAMPING.value, ProcessingType.COATING.value, ProcessingType.ASSEMBLY.value],
        "combined_ef": Decimal("420"),
        "applicable_categories": [ProductCategory.METALS_NON_FERROUS.value],
    },
    "plastic_packaging": {
        "description": "Injection Molding -> Assembly",
        "steps": [ProcessingType.INJECTION_MOLDING.value, ProcessingType.ASSEMBLY.value],
        "combined_ef": Decimal("565"),
        "applicable_categories": [ProductCategory.PLASTICS_THERMOPLASTIC.value, ProductCategory.PLASTICS_THERMOSET.value],
    },
    "semiconductor_chips": {
        "description": "Chemical Reaction -> Assembly -> Testing",
        "steps": [ProcessingType.CHEMICAL_REACTION.value, ProcessingType.ASSEMBLY.value],
        "combined_ef": Decimal("1800"),
        "applicable_categories": [ProductCategory.ELECTRONICS.value],
    },
    "food_products": {
        "description": "Milling -> Drying -> Packaging",
        "steps": [ProcessingType.MILLING.value, ProcessingType.DRYING.value, ProcessingType.ASSEMBLY.value],
        "combined_ef": Decimal("350"),
        "applicable_categories": [ProductCategory.FOOD_INGREDIENTS.value, ProductCategory.AGRICULTURAL.value],
    },
    "textile_garments": {
        "description": "Weaving -> Dyeing -> Assembly",
        "steps": [ProcessingType.TEXTILE_FINISHING.value, ProcessingType.COATING.value, ProcessingType.ASSEMBLY.value],
        "combined_ef": Decimal("620"),
        "applicable_categories": [ProductCategory.TEXTILES.value],
    },
    "glass_bottles": {
        "description": "Heat Treatment -> Coating",
        "steps": [ProcessingType.HEAT_TREATMENT.value, ProcessingType.COATING.value],
        "combined_ef": Decimal("500"),
        "applicable_categories": [ProductCategory.GLASS_CERAMICS.value],
    },
    "paper_products": {
        "description": "Milling -> Drying -> Coating",
        "steps": [ProcessingType.MILLING.value, ProcessingType.DRYING.value, ProcessingType.COATING.value],
        "combined_ef": Decimal("380"),
        "applicable_categories": [ProductCategory.WOOD_PAPER.value],
    },
}

# Sector benchmark factors (kgCO2e/tonne processed)
# Derived from cross-referencing industry averages with DEFRA/EPA benchmarks
SECTOR_BENCHMARK_FACTORS: Dict[str, Decimal] = {
    ProductCategory.METALS_FERROUS.value: Decimal("310"),
    ProductCategory.METALS_NON_FERROUS.value: Decimal("420"),
    ProductCategory.PLASTICS_THERMOPLASTIC.value: Decimal("580"),
    ProductCategory.PLASTICS_THERMOSET.value: Decimal("500"),
    ProductCategory.CHEMICALS.value: Decimal("750"),
    ProductCategory.FOOD_INGREDIENTS.value: Decimal("150"),
    ProductCategory.TEXTILES.value: Decimal("390"),
    ProductCategory.ELECTRONICS.value: Decimal("1050"),
    ProductCategory.GLASS_CERAMICS.value: Decimal("640"),
    ProductCategory.WOOD_PAPER.value: Decimal("210"),
    ProductCategory.MINERALS.value: Decimal("280"),
    ProductCategory.AGRICULTURAL.value: Decimal("125"),
}

# Sector benchmark uncertainty (always higher than process EF uncertainty)
SECTOR_UNCERTAINTY_PCT: Dict[str, Decimal] = {
    ProductCategory.METALS_FERROUS.value: Decimal("35"),
    ProductCategory.METALS_NON_FERROUS.value: Decimal("35"),
    ProductCategory.PLASTICS_THERMOPLASTIC.value: Decimal("40"),
    ProductCategory.PLASTICS_THERMOSET.value: Decimal("40"),
    ProductCategory.CHEMICALS.value: Decimal("45"),
    ProductCategory.FOOD_INGREDIENTS.value: Decimal("30"),
    ProductCategory.TEXTILES.value: Decimal("40"),
    ProductCategory.ELECTRONICS.value: Decimal("45"),
    ProductCategory.GLASS_CERAMICS.value: Decimal("35"),
    ProductCategory.WOOD_PAPER.value: Decimal("30"),
    ProductCategory.MINERALS.value: Decimal("35"),
    ProductCategory.AGRICULTURAL.value: Decimal("30"),
}

# Default DQI scores for average-data methods (scale 1-5, lower is better)
# PRD Section 5.8 medium-quality tier
_DQI_PROCESS_EF: Dict[str, Decimal] = {
    "reliability": Decimal("3"),
    "completeness": Decimal("3"),
    "temporal": Decimal("2"),
    "geographical": Decimal("3"),
    "technological": Decimal("3"),
}

_DQI_ENERGY_INTENSITY: Dict[str, Decimal] = {
    "reliability": Decimal("3"),
    "completeness": Decimal("3"),
    "temporal": Decimal("3"),
    "geographical": Decimal("3"),
    "technological": Decimal("2"),
}

_DQI_SECTOR_BENCHMARK: Dict[str, Decimal] = {
    "reliability": Decimal("4"),
    "completeness": Decimal("2"),
    "temporal": Decimal("3"),
    "geographical": Decimal("3"),
    "technological": Decimal("4"),
}

# Composite DQI scores by method (weighted average)
_DQI_COMPOSITE_PROCESS_EF: int = 55
_DQI_COMPOSITE_ENERGY_INTENSITY: int = 50
_DQI_COMPOSITE_SECTOR_BENCHMARK: int = 45

# Uncertainty ranges by method (PRD Section 5.9)
_UNCERTAINTY_PROCESS_PCT: Decimal = Decimal("25")
_UNCERTAINTY_ENERGY_PCT: Decimal = Decimal("30")
_UNCERTAINTY_SECTOR_PCT: Decimal = Decimal("35")
_UNCERTAINTY_CHAIN_PCT: Decimal = Decimal("30")

# Method-level uncertainty bounds
_UNCERTAINTY_BOUNDS: Dict[str, Tuple[Decimal, Decimal, Decimal]] = {
    AverageDataMethod.PROCESS_EF.value: (Decimal("15"), Decimal("25"), Decimal("40")),
    AverageDataMethod.ENERGY_INTENSITY.value: (Decimal("20"), Decimal("30"), Decimal("45")),
    AverageDataMethod.SECTOR_BENCHMARK.value: (Decimal("20"), Decimal("35"), Decimal("50")),
    AverageDataMethod.CHAIN.value: (Decimal("18"), Decimal("30"), Decimal("45")),
}

# Maximum batch size per calculation invocation
_MAX_BATCH_SIZE: int = 100_000

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

class DataQualityScore:
    """Data quality indicator score for an average-data calculation.

    Implements the 5-dimension DQI scoring per GHG Protocol Scope 3
    Standard Chapter 7, with a composite score on a 0-100 scale.

    Attributes:
        reliability: Reliability dimension score (1-5).
        completeness: Completeness dimension score (1-5).
        temporal: Temporal correlation score (1-5).
        geographical: Geographical correlation score (1-5).
        technological: Technological correlation score (1-5).
        composite: Weighted composite score (0-100).
        method: Calculation method that produced this score.
    """

    __slots__ = (
        "reliability", "completeness", "temporal",
        "geographical", "technological", "composite",
        "method",
    )

    def __init__(
        self,
        reliability: Decimal,
        completeness: Decimal,
        temporal: Decimal,
        geographical: Decimal,
        technological: Decimal,
        composite: int,
        method: str,
    ) -> None:
        """Initialize DataQualityScore.

        Args:
            reliability: Reliability dimension (1-5).
            completeness: Completeness dimension (1-5).
            temporal: Temporal dimension (1-5).
            geographical: Geographical dimension (1-5).
            technological: Technological dimension (1-5).
            composite: Composite score 0-100.
            method: Calculation method name.
        """
        self.reliability = reliability
        self.completeness = completeness
        self.temporal = temporal
        self.geographical = geographical
        self.technological = technological
        self.composite = composite
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary with all DQI dimension scores and composite.
        """
        return {
            "reliability": str(self.reliability),
            "completeness": str(self.completeness),
            "temporal": str(self.temporal),
            "geographical": str(self.geographical),
            "technological": str(self.technological),
            "composite": self.composite,
            "method": self.method,
        }

class UncertaintyResult:
    """Uncertainty quantification result for emission calculations.

    Provides lower and upper bounds at a 95% confidence interval,
    the default uncertainty percentage, and the method used.

    Attributes:
        emissions_kgco2e: Central estimate in kgCO2e.
        lower_bound_kgco2e: Lower bound at 95% CI.
        upper_bound_kgco2e: Upper bound at 95% CI.
        uncertainty_pct: Uncertainty percentage (symmetric).
        confidence_level: Confidence level (default 95).
        method: Method identifier.
    """

    __slots__ = (
        "emissions_kgco2e", "lower_bound_kgco2e", "upper_bound_kgco2e",
        "uncertainty_pct", "confidence_level", "method",
    )

    def __init__(
        self,
        emissions_kgco2e: Decimal,
        lower_bound_kgco2e: Decimal,
        upper_bound_kgco2e: Decimal,
        uncertainty_pct: Decimal,
        confidence_level: int = 95,
        method: str = "",
    ) -> None:
        """Initialize UncertaintyResult.

        Args:
            emissions_kgco2e: Central emission estimate.
            lower_bound_kgco2e: Lower bound.
            upper_bound_kgco2e: Upper bound.
            uncertainty_pct: Uncertainty percentage.
            confidence_level: Confidence level.
            method: Method identifier.
        """
        self.emissions_kgco2e = emissions_kgco2e
        self.lower_bound_kgco2e = lower_bound_kgco2e
        self.upper_bound_kgco2e = upper_bound_kgco2e
        self.uncertainty_pct = uncertainty_pct
        self.confidence_level = confidence_level
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the uncertainty result.
        """
        return {
            "emissions_kgco2e": str(self.emissions_kgco2e),
            "lower_bound_kgco2e": str(self.lower_bound_kgco2e),
            "upper_bound_kgco2e": str(self.upper_bound_kgco2e),
            "uncertainty_pct": str(self.uncertainty_pct),
            "confidence_level": self.confidence_level,
            "method": self.method,
        }

class ProductBreakdown:
    """Per-product emission breakdown from average-data calculation.

    Contains the detailed breakdown of emissions for a single
    intermediate product including the method used, emission factors
    applied, DQI scoring, and uncertainty range.

    Attributes:
        product_id: Unique product identifier.
        product_name: Human-readable product name.
        category: Product category enum value.
        quantity_tonnes: Quantity sold in tonnes.
        method: Calculation method used.
        emission_factor: Emission factor applied (kgCO2e/tonne).
        emission_factor_source: Source of the emission factor.
        emissions_kgco2e: Total emissions in kgCO2e.
        emissions_tco2e: Total emissions in tCO2e.
        processing_type: Processing type if applicable.
        chain_type: Chain type if multi-step chain was used.
        chain_steps: Chain steps if multi-step chain was used.
        energy_intensity_kwh_per_t: Energy intensity if energy method.
        grid_ef_kgco2e_per_kwh: Grid EF if energy method.
        countries: Countries involved in processing.
        dqi: Data quality indicator score.
        uncertainty: Uncertainty quantification.
        provenance_hash: SHA-256 provenance hash.
        calculated_at: Timestamp of calculation.
    """

    __slots__ = (
        "product_id", "product_name", "category",
        "quantity_tonnes", "method", "emission_factor",
        "emission_factor_source", "emissions_kgco2e",
        "emissions_tco2e", "processing_type", "chain_type",
        "chain_steps", "energy_intensity_kwh_per_t",
        "grid_ef_kgco2e_per_kwh", "countries", "dqi",
        "uncertainty", "provenance_hash", "calculated_at",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ProductBreakdown from keyword arguments.

        Args:
            **kwargs: Field values matching __slots__.
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the product breakdown.
        """
        result: Dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot, None)
            if val is not None:
                if isinstance(val, Decimal):
                    result[slot] = str(val)
                elif isinstance(val, datetime):
                    result[slot] = val.isoformat()
                elif hasattr(val, "to_dict"):
                    result[slot] = val.to_dict()
                elif isinstance(val, list):
                    result[slot] = val
                else:
                    result[slot] = val
        return result

class CalculationResult:
    """Aggregated calculation result from average-data engine.

    Contains the total emissions, per-product breakdowns, DQI,
    uncertainty, provenance, and processing metadata.

    Attributes:
        calculation_id: Unique calculation identifier.
        org_id: Organization identifier.
        reporting_year: Reporting year.
        method: Average-data sub-method used.
        total_emissions_kgco2e: Total emissions in kgCO2e.
        total_emissions_tco2e: Total emissions in tCO2e.
        product_count: Number of products calculated.
        breakdowns: Per-product breakdowns.
        dqi: Portfolio-level DQI score.
        uncertainty: Portfolio-level uncertainty.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Calculation duration in milliseconds.
        calculated_at: Timestamp of calculation.
        warnings: Any warnings generated.
        errors: Any errors encountered.
    """

    __slots__ = (
        "calculation_id", "org_id", "reporting_year",
        "method", "total_emissions_kgco2e",
        "total_emissions_tco2e", "product_count",
        "breakdowns", "dqi", "uncertainty",
        "provenance_hash", "processing_time_ms",
        "calculated_at", "warnings", "errors",
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize CalculationResult from keyword arguments.

        Args:
            **kwargs: Field values matching __slots__.
        """
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the calculation result.
        """
        result: Dict[str, Any] = {}
        for slot in self.__slots__:
            val = getattr(self, slot, None)
            if val is not None:
                if isinstance(val, Decimal):
                    result[slot] = str(val)
                elif isinstance(val, datetime):
                    result[slot] = val.isoformat()
                elif hasattr(val, "to_dict"):
                    result[slot] = val.to_dict()
                elif isinstance(val, list):
                    result[slot] = [
                        item.to_dict() if hasattr(item, "to_dict") else item
                        for item in val
                    ]
                else:
                    result[slot] = val
        return result

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the configured precision.

    Uses ROUND_HALF_UP rounding mode for consistency with GHG Protocol
    reporting conventions.

    Args:
        value: Raw Decimal value.

    Returns:
        Decimal quantized to DECIMAL_PLACES places using ROUND_HALF_UP.
    """
    try:
        return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)
    except (InvalidOperation, OverflowError):
        logger.warning("Quantize failed for value=%s, returning ZERO", value)
        return ZERO

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values, returning default on zero division.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when denominator is zero.

    Returns:
        Quantized quotient or default.
    """
    if denominator == ZERO or denominator is None:
        return default
    return _quantize(numerator / denominator)

def _compute_sha256(data: Any) -> str:
    """Compute SHA-256 hex digest for arbitrary data.

    Serializes the data to a canonical JSON string, encodes as UTF-8,
    and returns the hexadecimal SHA-256 digest.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character hexadecimal SHA-256 digest string.
    """
    try:
        canonical = json.dumps(
            data,
            sort_keys=True,
            default=str,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        logger.warning("SHA-256 hashing failed: %s", exc)
        return hashlib.sha256(b"fallback").hexdigest()

def _validate_quantity(quantity: Any, field_name: str = "quantity") -> Decimal:
    """Validate and coerce a quantity value to a positive Decimal.

    Args:
        quantity: The quantity value to validate.
        field_name: Name of the field for error messages.

    Returns:
        Positive Decimal value.

    Raises:
        ValueError: If quantity is None, negative, or non-numeric.
    """
    if quantity is None:
        raise ValueError(f"{field_name} must not be None")
    try:
        dec_val = Decimal(str(quantity))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a valid number, got {quantity!r}"
        ) from exc
    if dec_val < ZERO:
        raise ValueError(
            f"{field_name} must be non-negative, got {dec_val}"
        )
    return dec_val

def _resolve_category(category: Any) -> str:
    """Resolve a product category to its string key.

    Args:
        category: ProductCategory enum or string.

    Returns:
        Category string key.

    Raises:
        ValueError: If category is not recognized.
    """
    if isinstance(category, ProductCategory):
        return category.value
    cat_str = str(category).upper().strip()
    # Check if it matches a known category
    for pc in ProductCategory:
        if pc.value == cat_str or pc.name == cat_str:
            return pc.value
    raise ValueError(
        f"Unknown product category: {category!r}. "
        f"Valid categories: {[pc.value for pc in ProductCategory]}"
    )

def _resolve_processing_type(processing_type: Any) -> str:
    """Resolve a processing type to its string key.

    Args:
        processing_type: ProcessingType enum or string.

    Returns:
        Processing type string key.

    Raises:
        ValueError: If processing type is not recognized.
    """
    if isinstance(processing_type, ProcessingType):
        return processing_type.value
    pt_str = str(processing_type).upper().strip()
    for pt in ProcessingType:
        if pt.value == pt_str or pt.name == pt_str:
            return pt.value
    raise ValueError(
        f"Unknown processing type: {processing_type!r}. "
        f"Valid types: {[pt.value for pt in ProcessingType]}"
    )

# ===========================================================================
# AverageDataCalculatorEngine -- Thread-Safe Singleton
# ===========================================================================

class AverageDataCalculatorEngine:
    """Engine 3: Average-data emission calculator for Processing of Sold Products.

    Implements the average-data calculation method from the GHG Protocol
    Scope 3 Technical Guidance Chapter 6 (Category 10).  Emissions are
    computed by multiplying quantities of intermediate products sold by
    industry-average processing emission factors from recognized databases.

    This engine handles:

    1. **Process-Specific EF** -- Uses product-category-level processing
       emission factors (12 categories, kgCO2e/tonne).

    2. **Energy Intensity** -- Converts processing energy intensity
       (18 types, kWh/tonne) to emissions via grid emission factors
       (16 countries/regions, kgCO2e/kWh).

    3. **Sector Benchmark** -- Uses sector-level benchmark EFs that
       are broader and less specific than process EFs.

    4. **Multi-Step Chains** -- Applies combined EFs for 8 standard
       multi-step processing chains (e.g., stamping -> welding ->
       coating for steel automotive parts).

    5. **DQI Scoring** -- 5-dimension scoring with composite scores:
       process_ef=55, energy_intensity=50, sector_benchmark=45.

    6. **Uncertainty** -- Method-specific uncertainty ranges:
       process=+/-25%, energy=+/-30%, sector=+/-35%.

    7. **Provenance** -- SHA-256 hashing of all inputs and outputs.

    Thread Safety:
        Singleton via ``__new__`` with ``threading.RLock()``.

    Zero-Hallucination:
        All arithmetic uses ``Decimal``.  No LLM involvement in any
        numeric path.

    Attributes:
        _calculation_count: Total calculations performed.
        _total_emissions_kgco2e: Cumulative emissions calculated.
        _error_count: Total errors encountered.
        _last_calculation_time: Timestamp of last calculation.

    Example:
        >>> engine = AverageDataCalculatorEngine()
        >>> products = [{
        ...     "product_id": "P001",
        ...     "product_name": "Steel Coil",
        ...     "category": "METALS_FERROUS",
        ...     "quantity_tonnes": 500,
        ... }]
        >>> result = engine.calculate_process_ef(
        ...     products, org_id="ORG-1", reporting_year=2024,
        ... )
        >>> result.total_emissions_kgco2e
        Decimal('140000.00000000')
    """

    _instance: Optional[AverageDataCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    def __new__(cls) -> AverageDataCalculatorEngine:
        """Return the singleton instance, creating it on first call.

        Uses double-checked locking with ``threading.RLock`` to ensure
        thread-safe initialization. Only one instance is created for
        the lifetime of the process.

        Returns:
            The singleton AverageDataCalculatorEngine instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the engine with internal counters.

        Guarded by the ``_initialized`` class flag so repeated calls
        to ``__init__`` (from repeated instantiation) do not reset
        internal state.
        """
        if self.__class__._initialized:
            return
        with self._lock:
            if self.__class__._initialized:
                return
            self._calculation_count: int = 0
            self._batch_count: int = 0
            self._total_emissions_kgco2e: Decimal = ZERO
            self._error_count: int = 0
            self._last_calculation_time: Optional[datetime] = None
            self.__class__._initialized = True
            logger.info(
                "AverageDataCalculatorEngine initialized "
                "(agent=%s, version=%s, precision=%d, "
                "product_categories=%d, processing_types=%d, "
                "grid_regions=%d, chains=%d)",
                AGENT_ID,
                VERSION,
                DECIMAL_PLACES,
                len(PROCESSING_EMISSION_FACTORS),
                len(ENERGY_INTENSITY_FACTORS),
                len(GRID_EMISSION_FACTORS),
                len(PROCESSING_CHAINS),
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton for testing purposes.

        Clears the singleton instance and the initialized flag so
        that the next instantiation creates a fresh engine.  This
        method is intended for use in test fixtures only.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ------------------------------------------------------------------
    # Public API: Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return engine health status for operational monitoring.

        Returns:
            Dictionary with engine status, counters, and configuration.
        """
        return {
            "engine": "AverageDataCalculatorEngine",
            "agent_id": AGENT_ID,
            "version": VERSION,
            "status": "healthy",
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "total_emissions_kgco2e": str(self._total_emissions_kgco2e),
            "error_count": self._error_count,
            "last_calculation_time": (
                self._last_calculation_time.isoformat()
                if self._last_calculation_time else None
            ),
            "product_categories": len(PROCESSING_EMISSION_FACTORS),
            "processing_types": len(ENERGY_INTENSITY_FACTORS),
            "grid_regions": len(GRID_EMISSION_FACTORS),
            "chains": len(PROCESSING_CHAINS),
        }

    # ------------------------------------------------------------------
    # Public API: Main Dispatcher
    # ------------------------------------------------------------------

    def calculate(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
        method: str = AverageDataMethod.PROCESS_EF.value,
        countries: Optional[List[str]] = None,
    ) -> CalculationResult:
        """Dispatch to the appropriate average-data sub-method.

        This is the main entry point for average-data calculations.
        It routes to the specific sub-method based on the ``method``
        parameter.

        Args:
            products: List of product dictionaries with at minimum:
                - product_id (str): Unique product identifier.
                - product_name (str): Human-readable name.
                - category (str): ProductCategory value.
                - quantity_tonnes (float/Decimal): Quantity sold.
                Optional fields depending on method:
                - processing_type (str): For energy_intensity method.
                - countries (List[str]): For energy_intensity method.
                - chain_type (str): For chain method.
            org_id: Organization identifier.
            reporting_year: Reporting period year.
            method: One of 'process_ef', 'energy_intensity',
                'sector_benchmark', or 'chain'.
            countries: Default country list for energy intensity
                calculations when not specified per product.

        Returns:
            CalculationResult with total emissions and breakdowns.

        Raises:
            ValueError: If method is not recognized or inputs invalid.
        """
        logger.info(
            "AverageDataCalculatorEngine.calculate called "
            "(method=%s, products=%d, org_id=%s, year=%d)",
            method, len(products), org_id, reporting_year,
        )

        method_str = str(method).lower().strip()

        if method_str == AverageDataMethod.PROCESS_EF.value:
            return self.calculate_process_ef(
                products, org_id, reporting_year,
            )
        elif method_str == AverageDataMethod.ENERGY_INTENSITY.value:
            return self.calculate_energy_intensity(
                products, org_id, reporting_year,
                countries=countries,
            )
        elif method_str == AverageDataMethod.SECTOR_BENCHMARK.value:
            return self.calculate_sector_benchmark(
                products, org_id, reporting_year,
            )
        elif method_str == AverageDataMethod.CHAIN.value:
            return self._calculate_chain_batch(
                products, org_id, reporting_year,
            )
        else:
            raise ValueError(
                f"Unknown average-data method: {method!r}. "
                f"Valid methods: {[m.value for m in AverageDataMethod]}"
            )

    # ------------------------------------------------------------------
    # Public API: Process EF Method (Formula D)
    # ------------------------------------------------------------------

    def calculate_process_ef(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """Calculate emissions using process-specific EFs (Formula D).

        E_cat10 = Sum_i( Q_sold_i x EF_process_i )

        Uses the 12 product category processing emission factors from
        PROCESSING_EMISSION_FACTORS. This is the primary average-data
        method and provides the best accuracy within this engine.

        Args:
            products: List of product dictionaries with category and
                quantity_tonnes fields.
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult with process-EF-based emissions.

        Raises:
            ValueError: If products list is empty or exceeds maximum,
                or if any product has invalid fields.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())

        try:
            self._validate_products_input(products)

            breakdowns: List[ProductBreakdown] = []
            total_kgco2e = ZERO
            warnings: List[str] = []
            errors: List[str] = []

            for product in products:
                try:
                    breakdown = self._calculate_process_ef_single(product)
                    breakdowns.append(breakdown)
                    total_kgco2e += breakdown.emissions_kgco2e
                except Exception as exc:
                    error_msg = (
                        f"Product {product.get('product_id', 'unknown')}: "
                        f"{str(exc)}"
                    )
                    errors.append(error_msg)
                    logger.warning(
                        "Process EF calculation failed for product: %s",
                        error_msg,
                    )

            total_tco2e = _quantize(total_kgco2e / ONE_THOUSAND)
            total_kgco2e = _quantize(total_kgco2e)

            dqi = self.compute_dqi_score(
                method=AverageDataMethod.PROCESS_EF.value,
            )
            uncertainty = self.compute_uncertainty(
                total_kgco2e,
                method=AverageDataMethod.PROCESS_EF.value,
            )

            provenance_hash = self._build_provenance(
                method=AverageDataMethod.PROCESS_EF.value,
                inputs={"products": [p.get("product_id") for p in products]},
                result={"total_kgco2e": str(total_kgco2e)},
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            result = CalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method=AverageDataMethod.PROCESS_EF.value,
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                product_count=len(breakdowns),
                breakdowns=breakdowns,
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=utcnow(),
                warnings=warnings,
                errors=errors,
            )

            self._update_counters(total_kgco2e, len(products))

            logger.info(
                "calculate_process_ef completed "
                "(calc_id=%s, products=%d, total_kgco2e=%s, "
                "total_tco2e=%s, elapsed_ms=%s)",
                calculation_id,
                len(breakdowns),
                str(total_kgco2e),
                str(total_tco2e),
                str(elapsed_ms),
            )

            return result

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "calculate_process_ef failed: %s", str(exc), exc_info=True,
            )
            raise

    def _calculate_process_ef_single(
        self,
        product: Dict[str, Any],
    ) -> ProductBreakdown:
        """Calculate process-EF emissions for a single product.

        Args:
            product: Product dictionary with category and quantity_tonnes.

        Returns:
            ProductBreakdown with per-product emissions.

        Raises:
            ValueError: If category or quantity is invalid.
        """
        product_id = str(product.get("product_id", str(uuid.uuid4())))
        product_name = str(product.get("product_name", ""))
        category = _resolve_category(product.get("category", ""))
        quantity = _validate_quantity(
            product.get("quantity_tonnes"), "quantity_tonnes",
        )

        ef = self._resolve_processing_ef(category)
        emissions_kgco2e = _quantize(quantity * ef)
        emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

        dqi = self.compute_dqi_score(
            method=AverageDataMethod.PROCESS_EF.value,
        )
        uncertainty = self.compute_uncertainty(
            emissions_kgco2e,
            method=AverageDataMethod.PROCESS_EF.value,
            category=category,
        )

        provenance_hash = self._build_provenance(
            method=AverageDataMethod.PROCESS_EF.value,
            inputs={
                "product_id": product_id,
                "category": category,
                "quantity_tonnes": str(quantity),
            },
            result={
                "ef": str(ef),
                "emissions_kgco2e": str(emissions_kgco2e),
            },
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=quantity,
            method=AverageDataMethod.PROCESS_EF.value,
            emission_factor=ef,
            emission_factor_source="PROCESSING_EMISSION_FACTORS",
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            processing_type=None,
            chain_type=None,
            chain_steps=None,
            energy_intensity_kwh_per_t=None,
            grid_ef_kgco2e_per_kwh=None,
            countries=None,
            dqi=dqi,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculated_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Public API: Energy Intensity Method (Formula E)
    # ------------------------------------------------------------------

    def calculate_energy_intensity(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
        countries: Optional[List[str]] = None,
    ) -> CalculationResult:
        """Calculate emissions using energy intensity method (Formula E).

        E_cat10 = Sum_i( Q_sold_i x EI_i x EF_grid_avg )

        Multiplies quantity by processing energy intensity (kWh/t) and
        the grid emission factor for the processing location. When
        multiple countries are involved, uses the weighted average grid EF.

        Args:
            products: List of product dictionaries with at minimum:
                - product_id, product_name, category, quantity_tonnes
                - processing_type (str): Processing type code.
                - countries (List[str], optional): ISO country codes.
            org_id: Organization identifier.
            reporting_year: Reporting period year.
            countries: Default country list when not specified per product.

        Returns:
            CalculationResult with energy-intensity-based emissions.

        Raises:
            ValueError: If products list is empty or inputs invalid.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())
        default_countries = countries or ["GLOBAL"]

        try:
            self._validate_products_input(products)

            breakdowns: List[ProductBreakdown] = []
            total_kgco2e = ZERO
            warnings: List[str] = []
            errors: List[str] = []

            for product in products:
                try:
                    breakdown = self._calculate_energy_intensity_single(
                        product, default_countries,
                    )
                    breakdowns.append(breakdown)
                    total_kgco2e += breakdown.emissions_kgco2e
                except Exception as exc:
                    error_msg = (
                        f"Product {product.get('product_id', 'unknown')}: "
                        f"{str(exc)}"
                    )
                    errors.append(error_msg)
                    logger.warning(
                        "Energy intensity calculation failed: %s",
                        error_msg,
                    )

            total_tco2e = _quantize(total_kgco2e / ONE_THOUSAND)
            total_kgco2e = _quantize(total_kgco2e)

            dqi = self.compute_dqi_score(
                method=AverageDataMethod.ENERGY_INTENSITY.value,
            )
            uncertainty = self.compute_uncertainty(
                total_kgco2e,
                method=AverageDataMethod.ENERGY_INTENSITY.value,
            )

            provenance_hash = self._build_provenance(
                method=AverageDataMethod.ENERGY_INTENSITY.value,
                inputs={
                    "products": [p.get("product_id") for p in products],
                    "default_countries": default_countries,
                },
                result={"total_kgco2e": str(total_kgco2e)},
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            result = CalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method=AverageDataMethod.ENERGY_INTENSITY.value,
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                product_count=len(breakdowns),
                breakdowns=breakdowns,
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=utcnow(),
                warnings=warnings,
                errors=errors,
            )

            self._update_counters(total_kgco2e, len(products))

            logger.info(
                "calculate_energy_intensity completed "
                "(calc_id=%s, products=%d, total_kgco2e=%s, "
                "elapsed_ms=%s)",
                calculation_id,
                len(breakdowns),
                str(total_kgco2e),
                str(elapsed_ms),
            )

            return result

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "calculate_energy_intensity failed: %s",
                str(exc), exc_info=True,
            )
            raise

    def _calculate_energy_intensity_single(
        self,
        product: Dict[str, Any],
        default_countries: List[str],
    ) -> ProductBreakdown:
        """Calculate energy-intensity emissions for a single product.

        Args:
            product: Product dictionary.
            default_countries: Fallback country list.

        Returns:
            ProductBreakdown with per-product emissions.

        Raises:
            ValueError: If processing type or quantity is invalid.
        """
        product_id = str(product.get("product_id", str(uuid.uuid4())))
        product_name = str(product.get("product_name", ""))
        category = _resolve_category(product.get("category", ""))
        quantity = _validate_quantity(
            product.get("quantity_tonnes"), "quantity_tonnes",
        )

        processing_type_raw = product.get("processing_type")
        if processing_type_raw is None:
            raise ValueError(
                f"Product {product_id} requires 'processing_type' for "
                f"energy intensity calculation"
            )
        processing_type = _resolve_processing_type(processing_type_raw)

        product_countries = product.get("countries") or default_countries

        energy_intensity = self._resolve_energy_intensity(processing_type)
        grid_ef = self._compute_average_grid_ef(product_countries)

        # E = Q x EI x EF_grid
        emissions_kgco2e = _quantize(quantity * energy_intensity * grid_ef)
        emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

        dqi = self.compute_dqi_score(
            method=AverageDataMethod.ENERGY_INTENSITY.value,
        )
        uncertainty = self.compute_uncertainty(
            emissions_kgco2e,
            method=AverageDataMethod.ENERGY_INTENSITY.value,
        )

        provenance_hash = self._build_provenance(
            method=AverageDataMethod.ENERGY_INTENSITY.value,
            inputs={
                "product_id": product_id,
                "category": category,
                "quantity_tonnes": str(quantity),
                "processing_type": processing_type,
                "countries": product_countries,
            },
            result={
                "energy_intensity": str(energy_intensity),
                "grid_ef": str(grid_ef),
                "emissions_kgco2e": str(emissions_kgco2e),
            },
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=quantity,
            method=AverageDataMethod.ENERGY_INTENSITY.value,
            emission_factor=_quantize(energy_intensity * grid_ef),
            emission_factor_source="ENERGY_INTENSITY_FACTORS x GRID_EMISSION_FACTORS",
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            processing_type=processing_type,
            chain_type=None,
            chain_steps=None,
            energy_intensity_kwh_per_t=energy_intensity,
            grid_ef_kgco2e_per_kwh=grid_ef,
            countries=product_countries,
            dqi=dqi,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculated_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Public API: Sector Benchmark Method
    # ------------------------------------------------------------------

    def calculate_sector_benchmark(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """Calculate emissions using sector benchmark EFs.

        E_cat10 = Sum_i( Q_sold_i x EF_sector_i )

        Uses broader sector-level benchmark EFs which are typically
        higher than process-specific EFs due to the inclusion of
        additional processing overhead and variability.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            reporting_year: Reporting period year.

        Returns:
            CalculationResult with sector-benchmark-based emissions.

        Raises:
            ValueError: If products list is empty or inputs invalid.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())

        try:
            self._validate_products_input(products)

            breakdowns: List[ProductBreakdown] = []
            total_kgco2e = ZERO
            warnings: List[str] = []
            errors: List[str] = []

            for product in products:
                try:
                    breakdown = self._calculate_sector_benchmark_single(product)
                    breakdowns.append(breakdown)
                    total_kgco2e += breakdown.emissions_kgco2e
                except Exception as exc:
                    error_msg = (
                        f"Product {product.get('product_id', 'unknown')}: "
                        f"{str(exc)}"
                    )
                    errors.append(error_msg)
                    logger.warning(
                        "Sector benchmark calculation failed: %s",
                        error_msg,
                    )

            total_tco2e = _quantize(total_kgco2e / ONE_THOUSAND)
            total_kgco2e = _quantize(total_kgco2e)

            dqi = self.compute_dqi_score(
                method=AverageDataMethod.SECTOR_BENCHMARK.value,
            )
            uncertainty = self.compute_uncertainty(
                total_kgco2e,
                method=AverageDataMethod.SECTOR_BENCHMARK.value,
            )

            provenance_hash = self._build_provenance(
                method=AverageDataMethod.SECTOR_BENCHMARK.value,
                inputs={"products": [p.get("product_id") for p in products]},
                result={"total_kgco2e": str(total_kgco2e)},
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            result = CalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method=AverageDataMethod.SECTOR_BENCHMARK.value,
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                product_count=len(breakdowns),
                breakdowns=breakdowns,
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=utcnow(),
                warnings=warnings,
                errors=errors,
            )

            self._update_counters(total_kgco2e, len(products))

            logger.info(
                "calculate_sector_benchmark completed "
                "(calc_id=%s, products=%d, total_kgco2e=%s, "
                "elapsed_ms=%s)",
                calculation_id,
                len(breakdowns),
                str(total_kgco2e),
                str(elapsed_ms),
            )

            return result

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "calculate_sector_benchmark failed: %s",
                str(exc), exc_info=True,
            )
            raise

    def _calculate_sector_benchmark_single(
        self,
        product: Dict[str, Any],
    ) -> ProductBreakdown:
        """Calculate sector-benchmark emissions for a single product.

        Args:
            product: Product dictionary.

        Returns:
            ProductBreakdown with per-product emissions.
        """
        product_id = str(product.get("product_id", str(uuid.uuid4())))
        product_name = str(product.get("product_name", ""))
        category = _resolve_category(product.get("category", ""))
        quantity = _validate_quantity(
            product.get("quantity_tonnes"), "quantity_tonnes",
        )

        ef = SECTOR_BENCHMARK_FACTORS.get(category)
        if ef is None:
            raise ValueError(
                f"No sector benchmark EF for category: {category}"
            )

        emissions_kgco2e = _quantize(quantity * ef)
        emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

        dqi = self.compute_dqi_score(
            method=AverageDataMethod.SECTOR_BENCHMARK.value,
        )
        uncertainty = self.compute_uncertainty(
            emissions_kgco2e,
            method=AverageDataMethod.SECTOR_BENCHMARK.value,
            category=category,
        )

        provenance_hash = self._build_provenance(
            method=AverageDataMethod.SECTOR_BENCHMARK.value,
            inputs={
                "product_id": product_id,
                "category": category,
                "quantity_tonnes": str(quantity),
            },
            result={
                "ef": str(ef),
                "emissions_kgco2e": str(emissions_kgco2e),
            },
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=quantity,
            method=AverageDataMethod.SECTOR_BENCHMARK.value,
            emission_factor=ef,
            emission_factor_source="SECTOR_BENCHMARK_FACTORS",
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            processing_type=None,
            chain_type=None,
            chain_steps=None,
            energy_intensity_kwh_per_t=None,
            grid_ef_kgco2e_per_kwh=None,
            countries=None,
            dqi=dqi,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculated_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Public API: Single Product Emissions
    # ------------------------------------------------------------------

    def calculate_product_emissions(
        self,
        product: Dict[str, Any],
        method: str = AverageDataMethod.PROCESS_EF.value,
        countries: Optional[List[str]] = None,
    ) -> ProductBreakdown:
        """Calculate emissions for a single product using specified method.

        Convenience method for single-product calculation that dispatches
        to the appropriate internal calculator based on the method.

        Args:
            product: Product dictionary.
            method: Calculation method to use.
            countries: Country list for energy intensity method.

        Returns:
            ProductBreakdown with per-product emissions.

        Raises:
            ValueError: If method or product fields are invalid.
        """
        method_str = str(method).lower().strip()

        if method_str == AverageDataMethod.PROCESS_EF.value:
            return self._calculate_process_ef_single(product)
        elif method_str == AverageDataMethod.ENERGY_INTENSITY.value:
            return self._calculate_energy_intensity_single(
                product, countries or ["GLOBAL"],
            )
        elif method_str == AverageDataMethod.SECTOR_BENCHMARK.value:
            return self._calculate_sector_benchmark_single(product)
        elif method_str == AverageDataMethod.CHAIN.value:
            chain_type = product.get("chain_type")
            if not chain_type:
                raise ValueError(
                    "Product must have 'chain_type' for chain method"
                )
            return self.calculate_chain_emissions(product, chain_type)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    # ------------------------------------------------------------------
    # Public API: Multi-Step Chain Calculation
    # ------------------------------------------------------------------

    def calculate_chain_emissions(
        self,
        product: Dict[str, Any],
        chain_type: str,
    ) -> ProductBreakdown:
        """Calculate emissions for a multi-step processing chain.

        Uses pre-defined combined emission factors for standard
        processing chains (e.g., stamping -> welding -> coating for
        steel automotive parts). The combined EF accounts for all
        processing steps in the chain.

        Args:
            product: Product dictionary.
            chain_type: Chain type key from PROCESSING_CHAINS.

        Returns:
            ProductBreakdown with chain-based emissions.

        Raises:
            ValueError: If chain_type is unknown or product is invalid.
        """
        chain_key = str(chain_type).lower().strip()
        chain_def = PROCESSING_CHAINS.get(chain_key)
        if chain_def is None:
            raise ValueError(
                f"Unknown processing chain: {chain_type!r}. "
                f"Valid chains: {list(PROCESSING_CHAINS.keys())}"
            )

        product_id = str(product.get("product_id", str(uuid.uuid4())))
        product_name = str(product.get("product_name", ""))
        category = _resolve_category(product.get("category", ""))
        quantity = _validate_quantity(
            product.get("quantity_tonnes"), "quantity_tonnes",
        )

        # Verify product category is applicable for this chain
        applicable = chain_def["applicable_categories"]
        if category not in applicable:
            logger.warning(
                "Product %s category %s not in chain %s applicable "
                "categories %s; proceeding with calculation",
                product_id, category, chain_key, applicable,
            )

        combined_ef = chain_def["combined_ef"]
        chain_steps = chain_def["steps"]
        chain_description = chain_def["description"]

        emissions_kgco2e = _quantize(quantity * combined_ef)
        emissions_tco2e = _quantize(emissions_kgco2e / ONE_THOUSAND)

        dqi = self.compute_dqi_score(
            method=AverageDataMethod.CHAIN.value,
        )
        uncertainty = self.compute_uncertainty(
            emissions_kgco2e,
            method=AverageDataMethod.CHAIN.value,
        )

        provenance_hash = self._build_provenance(
            method=AverageDataMethod.CHAIN.value,
            inputs={
                "product_id": product_id,
                "category": category,
                "quantity_tonnes": str(quantity),
                "chain_type": chain_key,
                "chain_steps": chain_steps,
            },
            result={
                "combined_ef": str(combined_ef),
                "emissions_kgco2e": str(emissions_kgco2e),
            },
        )

        return ProductBreakdown(
            product_id=product_id,
            product_name=product_name,
            category=category,
            quantity_tonnes=quantity,
            method=AverageDataMethod.CHAIN.value,
            emission_factor=combined_ef,
            emission_factor_source=f"PROCESSING_CHAINS[{chain_key}]",
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_tco2e,
            processing_type=None,
            chain_type=chain_key,
            chain_steps=chain_steps,
            energy_intensity_kwh_per_t=None,
            grid_ef_kgco2e_per_kwh=None,
            countries=None,
            dqi=dqi,
            uncertainty=uncertainty,
            provenance_hash=provenance_hash,
            calculated_at=utcnow(),
        )

    # ------------------------------------------------------------------
    # Public API: DQI Scoring
    # ------------------------------------------------------------------

    def compute_dqi_score(
        self,
        method: str = AverageDataMethod.PROCESS_EF.value,
        product: Optional[Dict[str, Any]] = None,
    ) -> DataQualityScore:
        """Compute the data quality indicator score for a method.

        Returns default DQI scores based on the average-data sub-method.
        Composite scores: process_ef=55, energy_intensity=50,
        sector_benchmark=45, chain=50.

        Args:
            method: Average-data sub-method.
            product: Optional product dict for product-specific scoring.

        Returns:
            DataQualityScore with 5 dimensions and composite.
        """
        method_str = str(method).lower().strip()

        if method_str == AverageDataMethod.PROCESS_EF.value:
            dims = _DQI_PROCESS_EF.copy()
            composite = _DQI_COMPOSITE_PROCESS_EF
        elif method_str == AverageDataMethod.ENERGY_INTENSITY.value:
            dims = _DQI_ENERGY_INTENSITY.copy()
            composite = _DQI_COMPOSITE_ENERGY_INTENSITY
        elif method_str == AverageDataMethod.SECTOR_BENCHMARK.value:
            dims = _DQI_SECTOR_BENCHMARK.copy()
            composite = _DQI_COMPOSITE_SECTOR_BENCHMARK
        elif method_str == AverageDataMethod.CHAIN.value:
            dims = _DQI_ENERGY_INTENSITY.copy()
            composite = _DQI_COMPOSITE_ENERGY_INTENSITY
        else:
            dims = _DQI_SECTOR_BENCHMARK.copy()
            composite = _DQI_COMPOSITE_SECTOR_BENCHMARK

        return DataQualityScore(
            reliability=dims["reliability"],
            completeness=dims["completeness"],
            temporal=dims["temporal"],
            geographical=dims["geographical"],
            technological=dims["technological"],
            composite=composite,
            method=method_str,
        )

    # ------------------------------------------------------------------
    # Public API: Uncertainty
    # ------------------------------------------------------------------

    def compute_uncertainty(
        self,
        emissions_kgco2e: Decimal,
        method: str = AverageDataMethod.PROCESS_EF.value,
        category: Optional[str] = None,
    ) -> UncertaintyResult:
        """Compute uncertainty range for calculated emissions.

        Uses method-specific default uncertainty percentages.
        When a product category is provided, category-specific
        uncertainty may override the method default.

        Default uncertainty: process=+/-25%, energy=+/-30%,
        sector=+/-35%, chain=+/-30%.

        Args:
            emissions_kgco2e: Central emission estimate in kgCO2e.
            method: Average-data sub-method.
            category: Optional product category for category-specific
                uncertainty.

        Returns:
            UncertaintyResult with bounds and percentage.
        """
        method_str = str(method).lower().strip()

        # Determine base uncertainty percentage
        if method_str == AverageDataMethod.PROCESS_EF.value:
            base_pct = _UNCERTAINTY_PROCESS_PCT
        elif method_str == AverageDataMethod.ENERGY_INTENSITY.value:
            base_pct = _UNCERTAINTY_ENERGY_PCT
        elif method_str == AverageDataMethod.SECTOR_BENCHMARK.value:
            base_pct = _UNCERTAINTY_SECTOR_PCT
        elif method_str == AverageDataMethod.CHAIN.value:
            base_pct = _UNCERTAINTY_CHAIN_PCT
        else:
            base_pct = _UNCERTAINTY_SECTOR_PCT

        # Category-specific override for process_ef and sector_benchmark
        if category:
            if method_str == AverageDataMethod.PROCESS_EF.value:
                cat_pct = CATEGORY_UNCERTAINTY_PCT.get(category)
                if cat_pct is not None:
                    base_pct = cat_pct
            elif method_str == AverageDataMethod.SECTOR_BENCHMARK.value:
                cat_pct = SECTOR_UNCERTAINTY_PCT.get(category)
                if cat_pct is not None:
                    base_pct = cat_pct

        # Compute bounds
        fraction = base_pct / ONE_HUNDRED
        lower = _quantize(emissions_kgco2e * (ONE - fraction))
        upper = _quantize(emissions_kgco2e * (ONE + fraction))

        return UncertaintyResult(
            emissions_kgco2e=emissions_kgco2e,
            lower_bound_kgco2e=lower,
            upper_bound_kgco2e=upper,
            uncertainty_pct=base_pct,
            confidence_level=95,
            method=method_str,
        )

    # ------------------------------------------------------------------
    # Internal: EF Resolution
    # ------------------------------------------------------------------

    def _resolve_processing_ef(self, category: str) -> Decimal:
        """Resolve the processing emission factor for a product category.

        Looks up the EF from the PROCESSING_EMISSION_FACTORS table.

        Args:
            category: Product category string key.

        Returns:
            Emission factor in kgCO2e/tonne.

        Raises:
            ValueError: If category is not in the EF table.
        """
        ef = PROCESSING_EMISSION_FACTORS.get(category)
        if ef is None:
            raise ValueError(
                f"No processing emission factor for category: "
                f"{category!r}. Available: "
                f"{list(PROCESSING_EMISSION_FACTORS.keys())}"
            )
        return ef

    def _resolve_energy_intensity(self, processing_type: str) -> Decimal:
        """Resolve the energy intensity for a processing type.

        Looks up the default energy intensity (kWh/tonne) from the
        ENERGY_INTENSITY_FACTORS table.

        Args:
            processing_type: Processing type string key.

        Returns:
            Energy intensity in kWh/tonne.

        Raises:
            ValueError: If processing type is not in the table.
        """
        ei = ENERGY_INTENSITY_FACTORS.get(processing_type)
        if ei is None:
            raise ValueError(
                f"No energy intensity for processing type: "
                f"{processing_type!r}. Available: "
                f"{list(ENERGY_INTENSITY_FACTORS.keys())}"
            )
        return ei

    def _compute_average_grid_ef(
        self,
        countries: List[str],
    ) -> Decimal:
        """Compute the average grid emission factor for a list of countries.

        If no country codes are recognized, falls back to the GLOBAL
        average. Uses a simple arithmetic mean when multiple countries
        are specified.

        Args:
            countries: List of ISO country codes.

        Returns:
            Average grid EF in kgCO2e/kWh.
        """
        if not countries:
            return GRID_EMISSION_FACTORS["GLOBAL"]

        efs: List[Decimal] = []
        for code in countries:
            code_upper = str(code).upper().strip()
            ef = GRID_EMISSION_FACTORS.get(code_upper)
            if ef is not None:
                efs.append(ef)
            else:
                logger.warning(
                    "Grid EF not found for country %s; skipping",
                    code_upper,
                )

        if not efs:
            logger.warning(
                "No grid EFs resolved for countries %s; "
                "using GLOBAL average",
                countries,
            )
            return GRID_EMISSION_FACTORS["GLOBAL"]

        total = sum(efs, ZERO)
        count = Decimal(str(len(efs)))
        return _quantize(total / count)

    # ------------------------------------------------------------------
    # Internal: Chain Batch Calculation
    # ------------------------------------------------------------------

    def _calculate_chain_batch(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        reporting_year: int,
    ) -> CalculationResult:
        """Calculate chain emissions for a batch of products.

        Each product must specify a ``chain_type`` field. Products
        without a chain_type will attempt to auto-select based on
        their product category.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            CalculationResult with chain-based emissions.
        """
        start_time = time.monotonic()
        calculation_id = str(uuid.uuid4())

        try:
            self._validate_products_input(products)

            breakdowns: List[ProductBreakdown] = []
            total_kgco2e = ZERO
            warnings: List[str] = []
            errors: List[str] = []

            for product in products:
                try:
                    chain_type = product.get("chain_type")
                    if not chain_type:
                        chain_type = self._auto_select_chain(
                            product.get("category", ""),
                        )
                    if not chain_type:
                        error_msg = (
                            f"Product {product.get('product_id', 'unknown')}: "
                            f"No chain_type and cannot auto-select"
                        )
                        errors.append(error_msg)
                        continue

                    breakdown = self.calculate_chain_emissions(
                        product, chain_type,
                    )
                    breakdowns.append(breakdown)
                    total_kgco2e += breakdown.emissions_kgco2e

                except Exception as exc:
                    error_msg = (
                        f"Product {product.get('product_id', 'unknown')}: "
                        f"{str(exc)}"
                    )
                    errors.append(error_msg)
                    logger.warning(
                        "Chain calculation failed: %s", error_msg,
                    )

            total_tco2e = _quantize(total_kgco2e / ONE_THOUSAND)
            total_kgco2e = _quantize(total_kgco2e)

            dqi = self.compute_dqi_score(method=AverageDataMethod.CHAIN.value)
            uncertainty = self.compute_uncertainty(
                total_kgco2e, method=AverageDataMethod.CHAIN.value,
            )

            provenance_hash = self._build_provenance(
                method=AverageDataMethod.CHAIN.value,
                inputs={"products": [p.get("product_id") for p in products]},
                result={"total_kgco2e": str(total_kgco2e)},
            )

            elapsed_ms = _quantize(
                Decimal(str((time.monotonic() - start_time) * 1000))
            )

            return CalculationResult(
                calculation_id=calculation_id,
                org_id=org_id,
                reporting_year=reporting_year,
                method=AverageDataMethod.CHAIN.value,
                total_emissions_kgco2e=total_kgco2e,
                total_emissions_tco2e=total_tco2e,
                product_count=len(breakdowns),
                breakdowns=breakdowns,
                dqi=dqi,
                uncertainty=uncertainty,
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms,
                calculated_at=utcnow(),
                warnings=warnings,
                errors=errors,
            )

        except Exception as exc:
            self._error_count += 1
            logger.error(
                "_calculate_chain_batch failed: %s",
                str(exc), exc_info=True,
            )
            raise

    def _auto_select_chain(self, category_raw: str) -> Optional[str]:
        """Auto-select a processing chain based on product category.

        Searches PROCESSING_CHAINS for a chain whose applicable
        categories include the given category.

        Args:
            category_raw: Product category string.

        Returns:
            Chain type key or None if no match found.
        """
        try:
            category = _resolve_category(category_raw)
        except ValueError:
            return None

        for chain_key, chain_def in PROCESSING_CHAINS.items():
            if category in chain_def["applicable_categories"]:
                return chain_key
        return None

    # ------------------------------------------------------------------
    # Internal: Provenance
    # ------------------------------------------------------------------

    def _build_provenance(
        self,
        method: str,
        inputs: Dict[str, Any],
        result: Dict[str, Any],
    ) -> str:
        """Build a SHA-256 provenance hash for a calculation.

        Combines the method, inputs, result, and agent metadata into
        a deterministic hash for audit trail purposes.

        Args:
            method: Calculation method identifier.
            inputs: Dictionary of input parameters.
            result: Dictionary of output values.

        Returns:
            64-character hexadecimal SHA-256 digest.
        """
        provenance_data = {
            "agent_id": AGENT_ID,
            "version": VERSION,
            "engine": "AverageDataCalculatorEngine",
            "method": method,
            "inputs": inputs,
            "result": result,
            "timestamp": utcnow().isoformat(),
        }
        return _compute_sha256(provenance_data)

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_products_input(
        self,
        products: List[Dict[str, Any]],
    ) -> None:
        """Validate the products list input.

        Args:
            products: List of product dictionaries.

        Raises:
            ValueError: If products list is None, empty, or too large.
            TypeError: If products is not a list.
        """
        if products is None:
            raise ValueError("Products list must not be None")
        if not isinstance(products, list):
            raise TypeError(
                f"Products must be a list, got {type(products).__name__}"
            )
        if len(products) == 0:
            raise ValueError("Products list must not be empty")
        if len(products) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Products list exceeds maximum size of {_MAX_BATCH_SIZE}, "
                f"got {len(products)}"
            )

    # ------------------------------------------------------------------
    # Internal: Counter Updates
    # ------------------------------------------------------------------

    def _update_counters(
        self,
        emissions_kgco2e: Decimal,
        product_count: int,
    ) -> None:
        """Update internal counters in a thread-safe manner.

        Args:
            emissions_kgco2e: Emissions to add to cumulative total.
            product_count: Number of products processed.
        """
        with self._lock:
            self._calculation_count += 1
            self._batch_count += product_count
            self._total_emissions_kgco2e += emissions_kgco2e
            self._last_calculation_time = utcnow()

    # ------------------------------------------------------------------
    # Public API: Aggregation Helpers
    # ------------------------------------------------------------------

    def aggregate_by_category(
        self,
        breakdowns: List[ProductBreakdown],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by product category.

        Groups product breakdowns by their category and sums the
        emissions for each group.

        Args:
            breakdowns: List of ProductBreakdown objects.

        Returns:
            Dictionary mapping category to total kgCO2e.
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            category = getattr(bd, "category", None)
            emissions = getattr(bd, "emissions_kgco2e", ZERO)
            if category and emissions:
                result[category] = _quantize(result[category] + emissions)
        return dict(result)

    def aggregate_by_processing_type(
        self,
        breakdowns: List[ProductBreakdown],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by processing type.

        Groups product breakdowns by their processing type and sums
        the emissions for each group.

        Args:
            breakdowns: List of ProductBreakdown objects.

        Returns:
            Dictionary mapping processing type to total kgCO2e.
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            pt = getattr(bd, "processing_type", None)
            emissions = getattr(bd, "emissions_kgco2e", ZERO)
            if pt and emissions:
                result[pt] = _quantize(result[pt] + emissions)
        return dict(result)

    def aggregate_by_chain(
        self,
        breakdowns: List[ProductBreakdown],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by chain type.

        Groups product breakdowns by their chain type and sums
        the emissions for each group.

        Args:
            breakdowns: List of ProductBreakdown objects.

        Returns:
            Dictionary mapping chain type to total kgCO2e.
        """
        result: Dict[str, Decimal] = defaultdict(lambda: ZERO)
        for bd in breakdowns:
            ct = getattr(bd, "chain_type", None)
            emissions = getattr(bd, "emissions_kgco2e", ZERO)
            if ct and emissions:
                result[ct] = _quantize(result[ct] + emissions)
        return dict(result)

    def get_processing_ef(self, category: str) -> Optional[Decimal]:
        """Get the processing emission factor for a category.

        Public accessor for the EF table, useful for API endpoints.

        Args:
            category: Product category.

        Returns:
            EF in kgCO2e/tonne or None if not found.
        """
        try:
            cat = _resolve_category(category)
            return PROCESSING_EMISSION_FACTORS.get(cat)
        except ValueError:
            return None

    def get_energy_intensity(
        self,
        processing_type: str,
    ) -> Optional[Decimal]:
        """Get the energy intensity for a processing type.

        Public accessor for the energy intensity table.

        Args:
            processing_type: Processing type code.

        Returns:
            Energy intensity in kWh/tonne or None if not found.
        """
        try:
            pt = _resolve_processing_type(processing_type)
            return ENERGY_INTENSITY_FACTORS.get(pt)
        except ValueError:
            return None

    def get_grid_ef(self, country_code: str) -> Optional[Decimal]:
        """Get the grid emission factor for a country.

        Public accessor for the grid EF table.

        Args:
            country_code: ISO country code.

        Returns:
            Grid EF in kgCO2e/kWh or None if not found.
        """
        return GRID_EMISSION_FACTORS.get(
            str(country_code).upper().strip()
        )

    def get_chain_definition(
        self,
        chain_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the processing chain definition.

        Public accessor for the chain definition table.

        Args:
            chain_type: Chain type key.

        Returns:
            Chain definition dictionary or None if not found.
        """
        return PROCESSING_CHAINS.get(
            str(chain_type).lower().strip()
        )

    def list_categories(self) -> List[str]:
        """List all available product categories.

        Returns:
            List of product category string values.
        """
        return [pc.value for pc in ProductCategory]

    def list_processing_types(self) -> List[str]:
        """List all available processing types.

        Returns:
            List of processing type string values.
        """
        return [pt.value for pt in ProcessingType]

    def list_chains(self) -> List[str]:
        """List all available processing chains.

        Returns:
            List of chain type keys.
        """
        return list(PROCESSING_CHAINS.keys())

    def list_grid_countries(self) -> List[str]:
        """List all available grid EF country codes.

        Returns:
            List of country code strings.
        """
        return list(GRID_EMISSION_FACTORS.keys())
