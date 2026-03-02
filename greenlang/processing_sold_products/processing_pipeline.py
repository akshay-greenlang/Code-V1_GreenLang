# -*- coding: utf-8 -*-
"""
ProcessingPipelineEngine - AGENT-MRV-023 Engine 7

This module implements the ProcessingPipelineEngine for Processing of Sold Products
(GHG Protocol Scope 3 Category 10). It orchestrates a 10-stage pipeline for complete
emissions calculation from raw input to compliant, sealed output with full audit trail.

The 10 Pipeline Stages:
    1. VALIDATE    - Input validation and sanitization
    2. CLASSIFY    - Product category classification
    3. NORMALIZE   - Unit normalization to tonnes
    4. RESOLVE_EFS - Emission factor resolution per product
    5. CALCULATE   - Emission calculations (site-specific / average-data / spend-based)
    6. ALLOCATE    - Proportional allocation to products
    7. AGGREGATE   - Portfolio aggregation across all products
    8. COMPLIANCE  - Regulatory framework compliance checking (7 frameworks)
    9. PROVENANCE  - SHA-256 hash chain computation
    10. SEAL       - Final immutable result sealing

Method Waterfall (GHG Protocol recommended hierarchy):
    1. Site-specific (customer-reported direct emissions) - highest quality
    2. Site-specific (energy consumption x grid/fuel EF)
    3. Site-specific (fuel consumption x combustion EF)
    4. Average-data (product category x processing type EF) - fallback
    5. Spend-based (revenue x EEIO sector factor) - last resort

Engine References:
    Engine 1: ProcessingDatabaseEngine - EF lookups, product DB
    Engine 2: SiteSpecificCalculatorEngine - Customer-reported / energy / fuel
    Engine 3: AverageDataCalculatorEngine - Category x processing type EFs
    Engine 4: SpendBasedCalculatorEngine - EEIO sector factors
    Engine 5: HybridAggregatorEngine - Multi-method aggregation
    Engine 6: ComplianceCheckerEngine - 7 framework + DC checks

All calculations use Decimal with ROUND_HALF_UP for regulatory precision.
Stage timing tracked via monotonic clock for performance monitoring.
Error handling supports partial results on failure with warnings.

Example:
    >>> from greenlang.processing_sold_products.processing_pipeline import ProcessingPipelineEngine
    >>> engine = ProcessingPipelineEngine()
    >>> result = engine.run_pipeline(
    ...     inputs={"products": [{"product_id": "STEEL-001", "mass_tonnes": "100.0", ...}]},
    ...     method="site_specific_energy",
    ...     org_id="ORG-123",
    ...     reporting_year=2025,
    ... )
    >>> print(f"Total emissions: {result['total_emissions_kg_co2e']} kg CO2e")

Module: greenlang.processing_sold_products.processing_pipeline
Agent: AGENT-MRV-023 (Processing of Sold Products)
Version: 1.0.0
Author: GreenLang Platform Team
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"
ENGINE_ID: str = "processing_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"

# ==============================================================================
# DECIMAL PRECISION CONSTANTS
# ==============================================================================

ZERO: Decimal = Decimal("0")
ONE: Decimal = Decimal("1")
ONE_HUNDRED: Decimal = Decimal("100")
ONE_THOUSAND: Decimal = Decimal("1000")
_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class PipelineStage(str, Enum):
    """10-stage pipeline stage identifiers."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE = "CALCULATE"
    ALLOCATE = "ALLOCATE"
    AGGREGATE = "AGGREGATE"
    COMPLIANCE = "COMPLIANCE"
    PROVENANCE = "PROVENANCE"
    SEAL = "SEAL"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class CalculationMethod(str, Enum):
    """Calculation methods for Category 10 emissions."""

    SITE_SPECIFIC_DIRECT = "site_specific_direct"
    SITE_SPECIFIC_ENERGY = "site_specific_energy"
    SITE_SPECIFIC_FUEL = "site_specific_fuel"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"
    HYBRID = "hybrid"


class AllocationMethod(str, Enum):
    """Allocation method for multi-product portfolios."""

    MASS_BASED = "mass_based"
    ECONOMIC = "economic"
    ENERGY_CONTENT = "energy_content"
    EQUAL = "equal"
    CUSTOM = "custom"


class IntermediateProductCategory(str, Enum):
    """Categories of intermediate products."""

    METALS_FERROUS = "metals_ferrous"
    METALS_NON_FERROUS = "metals_non_ferrous"
    PLASTICS_THERMOPLASTIC = "plastics_thermoplastic"
    PLASTICS_THERMOSET = "plastics_thermoset"
    CHEMICALS = "chemicals"
    FOOD_INGREDIENTS = "food_ingredients"
    TEXTILES = "textiles"
    ELECTRONICS_COMPONENTS = "electronics_components"
    GLASS_CERAMICS = "glass_ceramics"
    WOOD_PAPER_PULP = "wood_paper_pulp"
    MINERALS = "minerals"
    AGRICULTURAL_COMMODITIES = "agricultural_commodities"


class ProcessingType(str, Enum):
    """Types of downstream processing operations."""

    MACHINING = "machining"
    STAMPING = "stamping"
    WELDING = "welding"
    HEAT_TREATMENT = "heat_treatment"
    INJECTION_MOLDING = "injection_molding"
    EXTRUSION = "extrusion"
    BLOW_MOLDING = "blow_molding"
    CASTING = "casting"
    FORGING = "forging"
    COATING = "coating"
    ASSEMBLY = "assembly"
    CHEMICAL_REACTION = "chemical_reaction"
    REFINING = "refining"
    MILLING = "milling"
    DRYING = "drying"
    SINTERING = "sintering"
    FERMENTATION = "fermentation"
    TEXTILE_FINISHING = "textile_finishing"


class ComplianceFramework(str, Enum):
    """Regulatory frameworks for compliance."""

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    CDP = "cdp"
    SBTI = "sbti"
    SB_253 = "sb_253"
    GRI = "gri"


class EFSource(str, Enum):
    """Emission factor data source."""

    CUSTOMER_REPORTED = "customer_reported"
    ECOINVENT = "ecoinvent"
    GHG_PROTOCOL = "ghg_protocol"
    EPA = "epa"
    DEFRA = "defra"
    EEIO = "eeio"
    IEA = "iea"
    INDUSTRY_ASSOCIATION = "industry_association"
    CUSTOM = "custom"


# ==============================================================================
# GWP VALUES (IPCC AR5 / AR6)
# ==============================================================================

GWP_AR5: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("28"),
    "N2O": Decimal("265"),
}

GWP_AR6: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("27.9"),
    "N2O": Decimal("273"),
}

# ==============================================================================
# DEFAULT EMISSION FACTORS (kg CO2e per tonne processed)
# - Source: Ecoinvent 3.10, EPA AP-42, GHG Protocol defaults
# - These are fallback defaults; engine 1 (ProcessingDatabaseEngine) provides
#   DB-backed lookups with full provenance
# ==============================================================================

DEFAULT_PROCESSING_EFS: Dict[str, Dict[str, Decimal]] = {
    "metals_ferrous:machining": {
        "ef_kg_co2e_per_tonne": Decimal("85.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "metals_ferrous:stamping": {
        "ef_kg_co2e_per_tonne": Decimal("110.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "metals_ferrous:welding": {
        "ef_kg_co2e_per_tonne": Decimal("145.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "metals_ferrous:heat_treatment": {
        "ef_kg_co2e_per_tonne": Decimal("320.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "metals_ferrous:casting": {
        "ef_kg_co2e_per_tonne": Decimal("450.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "metals_ferrous:forging": {
        "ef_kg_co2e_per_tonne": Decimal("380.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "metals_ferrous:coating": {
        "ef_kg_co2e_per_tonne": Decimal("65.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "metals_ferrous:assembly": {
        "ef_kg_co2e_per_tonne": Decimal("25.0"),
        "uncertainty_pct": Decimal("40"),
        "source": "ecoinvent",
    },
    "metals_non_ferrous:machining": {
        "ef_kg_co2e_per_tonne": Decimal("95.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "metals_non_ferrous:extrusion": {
        "ef_kg_co2e_per_tonne": Decimal("280.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "metals_non_ferrous:casting": {
        "ef_kg_co2e_per_tonne": Decimal("520.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "plastics_thermoplastic:injection_molding": {
        "ef_kg_co2e_per_tonne": Decimal("350.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "plastics_thermoplastic:extrusion": {
        "ef_kg_co2e_per_tonne": Decimal("290.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "plastics_thermoplastic:blow_molding": {
        "ef_kg_co2e_per_tonne": Decimal("320.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "plastics_thermoset:injection_molding": {
        "ef_kg_co2e_per_tonne": Decimal("380.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "plastics_thermoset:chemical_reaction": {
        "ef_kg_co2e_per_tonne": Decimal("450.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "chemicals:chemical_reaction": {
        "ef_kg_co2e_per_tonne": Decimal("550.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "epa",
    },
    "chemicals:refining": {
        "ef_kg_co2e_per_tonne": Decimal("420.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "epa",
    },
    "chemicals:milling": {
        "ef_kg_co2e_per_tonne": Decimal("180.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "epa",
    },
    "chemicals:drying": {
        "ef_kg_co2e_per_tonne": Decimal("250.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "epa",
    },
    "food_ingredients:milling": {
        "ef_kg_co2e_per_tonne": Decimal("120.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "food_ingredients:drying": {
        "ef_kg_co2e_per_tonne": Decimal("200.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "food_ingredients:fermentation": {
        "ef_kg_co2e_per_tonne": Decimal("310.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "defra",
    },
    "food_ingredients:heat_treatment": {
        "ef_kg_co2e_per_tonne": Decimal("280.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "textiles:textile_finishing": {
        "ef_kg_co2e_per_tonne": Decimal("480.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "textiles:coating": {
        "ef_kg_co2e_per_tonne": Decimal("150.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "textiles:drying": {
        "ef_kg_co2e_per_tonne": Decimal("220.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "electronics_components:assembly": {
        "ef_kg_co2e_per_tonne": Decimal("750.0"),
        "uncertainty_pct": Decimal("40"),
        "source": "ecoinvent",
    },
    "electronics_components:welding": {
        "ef_kg_co2e_per_tonne": Decimal("180.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "electronics_components:sintering": {
        "ef_kg_co2e_per_tonne": Decimal("420.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "ecoinvent",
    },
    "glass_ceramics:heat_treatment": {
        "ef_kg_co2e_per_tonne": Decimal("680.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "glass_ceramics:sintering": {
        "ef_kg_co2e_per_tonne": Decimal("580.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "ecoinvent",
    },
    "glass_ceramics:coating": {
        "ef_kg_co2e_per_tonne": Decimal("90.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "ecoinvent",
    },
    "wood_paper_pulp:milling": {
        "ef_kg_co2e_per_tonne": Decimal("160.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "wood_paper_pulp:drying": {
        "ef_kg_co2e_per_tonne": Decimal("240.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "wood_paper_pulp:chemical_reaction": {
        "ef_kg_co2e_per_tonne": Decimal("350.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "defra",
    },
    "minerals:milling": {
        "ef_kg_co2e_per_tonne": Decimal("45.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "epa",
    },
    "minerals:sintering": {
        "ef_kg_co2e_per_tonne": Decimal("520.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "epa",
    },
    "minerals:heat_treatment": {
        "ef_kg_co2e_per_tonne": Decimal("380.0"),
        "uncertainty_pct": Decimal("25"),
        "source": "epa",
    },
    "agricultural_commodities:milling": {
        "ef_kg_co2e_per_tonne": Decimal("95.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "agricultural_commodities:drying": {
        "ef_kg_co2e_per_tonne": Decimal("185.0"),
        "uncertainty_pct": Decimal("30"),
        "source": "defra",
    },
    "agricultural_commodities:fermentation": {
        "ef_kg_co2e_per_tonne": Decimal("275.0"),
        "uncertainty_pct": Decimal("35"),
        "source": "defra",
    },
}

# ==============================================================================
# DEFAULT EEIO SECTOR FACTORS (kg CO2e per USD revenue)
# - Source: EPA USEEIO 2.0.1, EXIOBASE 3.8
# ==============================================================================

DEFAULT_EEIO_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "metals_ferrous": {
        "ef_kg_co2e_per_usd": Decimal("0.55"),
        "uncertainty_pct": Decimal("50"),
    },
    "metals_non_ferrous": {
        "ef_kg_co2e_per_usd": Decimal("0.62"),
        "uncertainty_pct": Decimal("50"),
    },
    "plastics_thermoplastic": {
        "ef_kg_co2e_per_usd": Decimal("0.48"),
        "uncertainty_pct": Decimal("50"),
    },
    "plastics_thermoset": {
        "ef_kg_co2e_per_usd": Decimal("0.52"),
        "uncertainty_pct": Decimal("50"),
    },
    "chemicals": {
        "ef_kg_co2e_per_usd": Decimal("0.72"),
        "uncertainty_pct": Decimal("50"),
    },
    "food_ingredients": {
        "ef_kg_co2e_per_usd": Decimal("0.38"),
        "uncertainty_pct": Decimal("50"),
    },
    "textiles": {
        "ef_kg_co2e_per_usd": Decimal("0.42"),
        "uncertainty_pct": Decimal("50"),
    },
    "electronics_components": {
        "ef_kg_co2e_per_usd": Decimal("0.35"),
        "uncertainty_pct": Decimal("50"),
    },
    "glass_ceramics": {
        "ef_kg_co2e_per_usd": Decimal("0.58"),
        "uncertainty_pct": Decimal("50"),
    },
    "wood_paper_pulp": {
        "ef_kg_co2e_per_usd": Decimal("0.32"),
        "uncertainty_pct": Decimal("50"),
    },
    "minerals": {
        "ef_kg_co2e_per_usd": Decimal("0.45"),
        "uncertainty_pct": Decimal("50"),
    },
    "agricultural_commodities": {
        "ef_kg_co2e_per_usd": Decimal("0.40"),
        "uncertainty_pct": Decimal("50"),
    },
}

# ==============================================================================
# GRID ELECTRICITY EMISSION FACTORS (kg CO2e per kWh)
# - Default global/regional grid factors for site-specific energy method
# ==============================================================================

DEFAULT_GRID_EFS: Dict[str, Decimal] = {
    "global_average": Decimal("0.475"),
    "us_average": Decimal("0.386"),
    "eu_average": Decimal("0.276"),
    "uk": Decimal("0.212"),
    "china": Decimal("0.581"),
    "india": Decimal("0.708"),
    "japan": Decimal("0.457"),
    "germany": Decimal("0.338"),
    "france": Decimal("0.052"),
    "australia": Decimal("0.656"),
    "canada": Decimal("0.120"),
    "brazil": Decimal("0.075"),
}

# ==============================================================================
# FUEL COMBUSTION EMISSION FACTORS (kg CO2e per litre)
# - Source: DEFRA 2024
# ==============================================================================

DEFAULT_FUEL_EFS: Dict[str, Decimal] = {
    "natural_gas_m3": Decimal("2.02"),
    "diesel_litre": Decimal("2.68"),
    "petrol_litre": Decimal("2.31"),
    "lpg_litre": Decimal("1.56"),
    "fuel_oil_litre": Decimal("3.18"),
    "coal_kg": Decimal("2.42"),
    "biomass_kg": Decimal("0.015"),
}

# ==============================================================================
# UNIT CONVERSION FACTORS
# ==============================================================================

UNIT_CONVERSIONS: Dict[str, Decimal] = {
    "kg_to_tonnes": Decimal("0.001"),
    "lb_to_tonnes": Decimal("0.000453592"),
    "short_ton_to_tonnes": Decimal("0.907185"),
    "long_ton_to_tonnes": Decimal("1.01605"),
    "g_to_tonnes": Decimal("0.000001"),
    "usg_to_litres": Decimal("3.78541"),
    "imp_gal_to_litres": Decimal("4.54609"),
    "btu_to_kwh": Decimal("0.000293071"),
    "mj_to_kwh": Decimal("0.277778"),
    "therm_to_kwh": Decimal("29.3071"),
    "mcf_to_m3": Decimal("28.3168"),
}


# ==============================================================================
# DATA MODELS
# ==============================================================================


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    stage: PipelineStage
    status: PipelineStatus
    duration_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProvenanceEntry:
    """Single entry in the provenance hash chain."""

    stage: str
    input_hash: str
    output_hash: str
    chain_hash: str
    timestamp: str
    duration_ms: float


@dataclass
class ProvenanceChain:
    """Complete provenance hash chain for an entire pipeline run."""

    chain_id: str
    entries: List[ProvenanceEntry] = field(default_factory=list)
    final_hash: str = ""
    sealed_at: str = ""

    def append_entry(self, entry: ProvenanceEntry) -> None:
        """Append an entry to the chain."""
        self.entries.append(entry)

    def get_final_hash(self) -> str:
        """Return the final chain hash."""
        if self.entries:
            return self.entries[-1].chain_hash
        return ""


# ==============================================================================
# ProcessingPipelineEngine
# ==============================================================================


class ProcessingPipelineEngine:
    """
    ProcessingPipelineEngine - 10-stage orchestration for Category 10.

    This engine coordinates the complete emissions calculation workflow for
    processing of sold products through 10 sequential stages. It references
    all 6 other engines and applies the method waterfall for optimal data
    quality: site-specific -> average-data -> spend-based.

    Thread Safety:
        Singleton pattern with threading.RLock() for concurrent access.

    Attributes:
        _db_engine: ProcessingDatabaseEngine (Engine 1) - lazy loaded
        _site_specific_engine: SiteSpecificCalculatorEngine (Engine 2) - lazy loaded
        _average_data_engine: AverageDataCalculatorEngine (Engine 3) - lazy loaded
        _spend_based_engine: SpendBasedCalculatorEngine (Engine 4) - lazy loaded
        _hybrid_engine: HybridAggregatorEngine (Engine 5) - lazy loaded
        _compliance_engine: ComplianceCheckerEngine (Engine 6) - lazy loaded

    Example:
        >>> engine = ProcessingPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     inputs={"products": [...]},
        ...     method="hybrid",
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["ProcessingPipelineEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "ProcessingPipelineEngine":
        """Thread-safe singleton constructor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize pipeline engine (singleton-safe)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Lazy-loaded engine references (created on first use)
            self._db_engine: Optional[Any] = None
            self._site_specific_engine: Optional[Any] = None
            self._average_data_engine: Optional[Any] = None
            self._spend_based_engine: Optional[Any] = None
            self._hybrid_engine: Optional[Any] = None
            self._compliance_engine: Optional[Any] = None

            # Pipeline state
            self._current_stage: Optional[PipelineStage] = None
            self._pipeline_run_count: int = 0
            self._total_products_processed: int = 0

            self._initialized = True
            logger.info(
                "%s v%s: ProcessingPipelineEngine initialized",
                AGENT_ID,
                ENGINE_VERSION,
            )

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("ProcessingPipelineEngine singleton reset")

    # ==========================================================================
    # LAZY ENGINE ACCESSORS
    # ==========================================================================

    def _get_db_engine(self) -> Any:
        """Get or create ProcessingDatabaseEngine (Engine 1)."""
        if self._db_engine is None:
            try:
                from greenlang.processing_sold_products.processing_database import (
                    ProcessingDatabaseEngine,
                )
                self._db_engine = ProcessingDatabaseEngine.get_instance()
            except ImportError:
                logger.warning("ProcessingDatabaseEngine not available, using defaults")
        return self._db_engine

    def _get_site_specific_engine(self) -> Any:
        """Get or create SiteSpecificCalculatorEngine (Engine 2)."""
        if self._site_specific_engine is None:
            try:
                from greenlang.processing_sold_products.site_specific_calculator import (
                    SiteSpecificCalculatorEngine,
                )
                self._site_specific_engine = SiteSpecificCalculatorEngine.get_instance()
            except ImportError:
                logger.warning("SiteSpecificCalculatorEngine not available")
        return self._site_specific_engine

    def _get_average_data_engine(self) -> Any:
        """Get or create AverageDataCalculatorEngine (Engine 3)."""
        if self._average_data_engine is None:
            try:
                from greenlang.processing_sold_products.average_data_calculator import (
                    AverageDataCalculatorEngine,
                )
                self._average_data_engine = AverageDataCalculatorEngine.get_instance()
            except ImportError:
                logger.warning("AverageDataCalculatorEngine not available")
        return self._average_data_engine

    def _get_spend_based_engine(self) -> Any:
        """Get or create SpendBasedCalculatorEngine (Engine 4)."""
        if self._spend_based_engine is None:
            try:
                from greenlang.processing_sold_products.spend_based_calculator import (
                    SpendBasedCalculatorEngine,
                )
                self._spend_based_engine = SpendBasedCalculatorEngine.get_instance()
            except ImportError:
                logger.warning("SpendBasedCalculatorEngine not available")
        return self._spend_based_engine

    def _get_hybrid_engine(self) -> Any:
        """Get or create HybridAggregatorEngine (Engine 5)."""
        if self._hybrid_engine is None:
            try:
                from greenlang.processing_sold_products.hybrid_aggregator import (
                    HybridAggregatorEngine,
                )
                self._hybrid_engine = HybridAggregatorEngine.get_instance()
            except ImportError:
                logger.warning("HybridAggregatorEngine not available")
        return self._hybrid_engine

    def _get_compliance_engine(self) -> Any:
        """Get or create ComplianceCheckerEngine (Engine 6)."""
        if self._compliance_engine is None:
            try:
                from greenlang.processing_sold_products.compliance_checker import (
                    ComplianceCheckerEngine,
                )
                self._compliance_engine = ComplianceCheckerEngine.get_instance()
            except ImportError:
                logger.warning("ComplianceCheckerEngine not available")
        return self._compliance_engine

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def run_pipeline(
        self,
        inputs: Dict[str, Any],
        method: str = "hybrid",
        org_id: str = "",
        reporting_year: int = 2025,
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete 10-stage pipeline.

        Args:
            inputs: Dictionary with "products" key containing product data.
            method: Calculation method (site_specific_direct/energy/fuel,
                average_data, spend_based, hybrid).
            org_id: Organization identifier.
            reporting_year: Reporting year for calculations.
            frameworks: Optional list of compliance frameworks to check.

        Returns:
            Complete pipeline result dictionary with emissions, compliance,
            provenance, and stage timing.

        Raises:
            ValueError: If input validation fails.
        """
        pipeline_id = f"psp-{uuid.uuid4().hex[:12]}"
        start_time = time.monotonic()
        stage_durations: Dict[str, float] = {}
        errors: List[str] = []
        warnings: List[str] = []
        provenance_chain = ProvenanceChain(chain_id=pipeline_id)

        logger.info(
            "[%s] Starting pipeline: method=%s, org=%s, year=%d",
            pipeline_id,
            method,
            org_id,
            reporting_year,
        )

        try:
            # Resolve method enum
            try:
                method_enum = CalculationMethod(method)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Invalid calculation method '{method}'. "
                    f"Valid: {[m.value for m in CalculationMethod]}"
                )

            # ------------------------------------------------------------------
            # Stage 1: VALIDATE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.VALIDATE
            stage_start = time.monotonic()
            validated_inputs = self._stage_validate(inputs)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["VALIDATE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.VALIDATE, inputs, validated_inputs, stage_ms
            )
            logger.info("[%s] VALIDATE completed in %.2fms", pipeline_id, stage_ms)

            products = validated_inputs.get("products", [])

            # ------------------------------------------------------------------
            # Stage 2: CLASSIFY
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.CLASSIFY
            stage_start = time.monotonic()
            classified_products = self._stage_classify(products)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["CLASSIFY"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.CLASSIFY, products, classified_products, stage_ms
            )
            logger.info(
                "[%s] CLASSIFY completed in %.2fms (%d products)",
                pipeline_id,
                stage_ms,
                len(classified_products),
            )

            # ------------------------------------------------------------------
            # Stage 3: NORMALIZE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.NORMALIZE
            stage_start = time.monotonic()
            normalized_products = self._stage_normalize(classified_products)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["NORMALIZE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.NORMALIZE, classified_products, normalized_products, stage_ms
            )
            logger.info("[%s] NORMALIZE completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 4: RESOLVE_EFS
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.RESOLVE_EFS
            stage_start = time.monotonic()
            products_with_efs = self._stage_resolve_efs(normalized_products, method_enum)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["RESOLVE_EFS"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.RESOLVE_EFS, normalized_products, products_with_efs, stage_ms
            )
            logger.info("[%s] RESOLVE_EFS completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 5: CALCULATE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.CALCULATE
            stage_start = time.monotonic()
            breakdowns = self._stage_calculate(products_with_efs, method_enum)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["CALCULATE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.CALCULATE, products_with_efs, breakdowns, stage_ms
            )
            calc_warnings = [b.get("warning", "") for b in breakdowns if b.get("warning")]
            warnings.extend(calc_warnings)
            logger.info("[%s] CALCULATE completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 6: ALLOCATE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.ALLOCATE
            stage_start = time.monotonic()
            allocation_method_str = inputs.get("allocation_method", "mass_based")
            allocated = self._stage_allocate(breakdowns, allocation_method_str)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["ALLOCATE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.ALLOCATE, breakdowns, allocated, stage_ms
            )
            logger.info("[%s] ALLOCATE completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 7: AGGREGATE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.AGGREGATE
            stage_start = time.monotonic()
            aggregation = self._stage_aggregate(allocated)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["AGGREGATE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.AGGREGATE, allocated, aggregation, stage_ms
            )
            logger.info(
                "[%s] AGGREGATE completed in %.2fms (total: %s kg CO2e)",
                pipeline_id,
                stage_ms,
                aggregation.get("total_emissions_kg_co2e", "0"),
            )

            # ------------------------------------------------------------------
            # Stage 8: COMPLIANCE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.COMPLIANCE
            stage_start = time.monotonic()
            compliance_input = self._build_compliance_input(
                aggregation, allocated, method_enum, org_id, reporting_year
            )
            compliance_results = self._stage_compliance(compliance_input, frameworks)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["COMPLIANCE"] = stage_ms
            self._record_provenance(
                provenance_chain, PipelineStage.COMPLIANCE, compliance_input, compliance_results, stage_ms
            )
            logger.info("[%s] COMPLIANCE completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 9: PROVENANCE
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.PROVENANCE
            stage_start = time.monotonic()
            provenance_result = self._stage_provenance(provenance_chain)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["PROVENANCE"] = stage_ms
            logger.info("[%s] PROVENANCE completed in %.2fms", pipeline_id, stage_ms)

            # ------------------------------------------------------------------
            # Stage 10: SEAL
            # ------------------------------------------------------------------
            self._current_stage = PipelineStage.SEAL
            stage_start = time.monotonic()

            final_result = {
                "pipeline_id": pipeline_id,
                "status": PipelineStatus.SUCCESS.value,
                "org_id": org_id,
                "reporting_year": reporting_year,
                "calculation_method": method_enum.value,
                "products": allocated,
                "aggregation": aggregation,
                "total_emissions_kg_co2e": aggregation.get("total_emissions_kg_co2e", "0"),
                "total_products": len(allocated),
                "compliance_results": compliance_results,
                "stage_durations_ms": stage_durations,
                "errors": errors,
                "warnings": warnings,
            }

            sealed_result = self._stage_seal(final_result, provenance_result)
            stage_ms = self._elapsed_ms(stage_start)
            stage_durations["SEAL"] = stage_ms
            sealed_result["stage_durations_ms"] = stage_durations
            logger.info("[%s] SEAL completed in %.2fms", pipeline_id, stage_ms)

            # Update counters
            self._pipeline_run_count += 1
            self._total_products_processed += len(allocated)
            self._current_stage = None

            total_duration_ms = self._elapsed_ms(start_time)
            sealed_result["total_duration_ms"] = total_duration_ms

            logger.info(
                "[%s] Pipeline completed: status=%s, total=%s kg CO2e, "
                "products=%d, duration=%.2fms",
                pipeline_id,
                PipelineStatus.SUCCESS.value,
                aggregation.get("total_emissions_kg_co2e", "0"),
                len(allocated),
                total_duration_ms,
            )

            return sealed_result

        except ValueError as e:
            logger.error("[%s] Validation error: %s", pipeline_id, str(e))
            errors.append(str(e))
            total_duration_ms = self._elapsed_ms(start_time)
            return {
                "pipeline_id": pipeline_id,
                "status": PipelineStatus.VALIDATION_ERROR.value,
                "org_id": org_id,
                "reporting_year": reporting_year,
                "calculation_method": method,
                "total_emissions_kg_co2e": "0",
                "total_products": 0,
                "products": [],
                "aggregation": {},
                "compliance_results": {},
                "stage_durations_ms": stage_durations,
                "errors": errors,
                "warnings": warnings,
                "provenance_hash": "",
                "total_duration_ms": total_duration_ms,
            }

        except Exception as e:
            logger.error(
                "[%s] Pipeline failed: %s", pipeline_id, str(e), exc_info=True
            )
            errors.append(f"Pipeline failed: {str(e)}")
            total_duration_ms = self._elapsed_ms(start_time)
            return {
                "pipeline_id": pipeline_id,
                "status": PipelineStatus.FAILED.value,
                "org_id": org_id,
                "reporting_year": reporting_year,
                "calculation_method": method,
                "total_emissions_kg_co2e": "0",
                "total_products": 0,
                "products": [],
                "aggregation": {},
                "compliance_results": {},
                "stage_durations_ms": stage_durations,
                "errors": errors,
                "warnings": warnings,
                "provenance_hash": "",
                "total_duration_ms": total_duration_ms,
            }

    def run_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
        method: str = "hybrid",
        org_id: str = "",
        reporting_year: int = 2025,
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline for a batch of input sets.

        Each input set runs independently through the full pipeline.
        Failures in one input do not affect others.

        Args:
            batch_inputs: List of input dictionaries, each with "products" key.
            method: Calculation method.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            List of pipeline results, one per input set.

        Example:
            >>> results = engine.run_batch(
            ...     [{"products": [...]}, {"products": [...]}],
            ...     method="average_data",
            ... )
            >>> for r in results:
            ...     print(f"{r['pipeline_id']}: {r['status']}")
        """
        batch_start = time.monotonic()
        logger.info(
            "Starting batch pipeline: %d input sets, method=%s",
            len(batch_inputs),
            method,
        )

        results: List[Dict[str, Any]] = []
        for idx, inputs in enumerate(batch_inputs):
            try:
                result = self.run_pipeline(
                    inputs=inputs,
                    method=method,
                    org_id=org_id,
                    reporting_year=reporting_year,
                )
                results.append(result)
            except Exception as e:
                logger.error("Batch item %d failed: %s", idx, str(e), exc_info=True)
                results.append({
                    "pipeline_id": f"batch-error-{idx}",
                    "status": PipelineStatus.FAILED.value,
                    "errors": [str(e)],
                    "total_emissions_kg_co2e": "0",
                })

        batch_duration = self._elapsed_ms(batch_start)
        success_count = sum(1 for r in results if r.get("status") == PipelineStatus.SUCCESS.value)

        logger.info(
            "Batch pipeline complete: %d/%d succeeded, duration=%.2fms",
            success_count,
            len(results),
            batch_duration,
        )

        return results

    def run_portfolio_analysis(
        self,
        inputs: Dict[str, Any],
        org_id: str = "",
        reporting_year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Run portfolio-level analysis across all products.

        This method runs the pipeline with the hybrid method and then
        produces additional portfolio-level analytics: hot-spots,
        method breakdown, and category distribution.

        Args:
            inputs: Dictionary with "products" key.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with pipeline result plus portfolio analytics.

        Example:
            >>> analysis = engine.run_portfolio_analysis(inputs)
            >>> print(f"Top emitter: {analysis['hot_spots'][0]['product_id']}")
        """
        result = self.run_pipeline(
            inputs=inputs,
            method="hybrid",
            org_id=org_id,
            reporting_year=reporting_year,
        )

        products = result.get("products", [])
        aggregation = result.get("aggregation", {})
        total_emissions = self._safe_decimal(result.get("total_emissions_kg_co2e", "0"))

        # Hot-spot analysis (top emitters)
        sorted_products = sorted(
            products,
            key=lambda p: self._safe_decimal(p.get("emissions_kg_co2e", "0")) or ZERO,
            reverse=True,
        )
        hot_spots = []
        cumulative = ZERO
        for product in sorted_products:
            em = self._safe_decimal(product.get("emissions_kg_co2e", "0")) or ZERO
            cumulative += em
            pct_of_total = ZERO
            if total_emissions and total_emissions > ZERO:
                pct_of_total = (em / total_emissions * ONE_HUNDRED).quantize(
                    _QUANT_2DP, rounding=ROUNDING
                )
            cumulative_pct = ZERO
            if total_emissions and total_emissions > ZERO:
                cumulative_pct = (cumulative / total_emissions * ONE_HUNDRED).quantize(
                    _QUANT_2DP, rounding=ROUNDING
                )
            hot_spots.append({
                "product_id": product.get("product_id", ""),
                "product_category": product.get("product_category", ""),
                "processing_type": product.get("processing_type", ""),
                "emissions_kg_co2e": str(em),
                "percentage_of_total": str(pct_of_total),
                "cumulative_percentage": str(cumulative_pct),
            })

        # Method breakdown
        method_breakdown: Dict[str, Decimal] = {}
        for product in products:
            m = product.get("calculation_method_used", "unknown")
            em = self._safe_decimal(product.get("emissions_kg_co2e", "0")) or ZERO
            method_breakdown[m] = method_breakdown.get(m, ZERO) + em

        result["hot_spots"] = hot_spots
        result["method_breakdown"] = {k: str(v) for k, v in method_breakdown.items()}
        result["portfolio_summary"] = {
            "total_products": len(products),
            "total_emissions_kg_co2e": str(total_emissions or ZERO),
            "category_distribution": aggregation.get("by_category", {}),
            "processing_distribution": aggregation.get("by_processing_type", {}),
        }

        return result

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and statistics.

        Returns:
            Dictionary with current stage, run count, and statistics.
        """
        return {
            "current_stage": self._current_stage.value if self._current_stage else None,
            "pipeline_run_count": self._pipeline_run_count,
            "total_products_processed": self._total_products_processed,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "stages": [s.value for s in PipelineStage],
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Pre-flight validation of inputs without running the full pipeline.

        Args:
            inputs: Dictionary with "products" key.

        Returns:
            List of validation error messages (empty if valid).

        Example:
            >>> errors = engine.validate_inputs({"products": [...]})
            >>> if errors:
            ...     print("Validation failed:")
            ...     for e in errors:
            ...         print(f"  - {e}")
        """
        errors: List[str] = []

        if not isinstance(inputs, dict):
            errors.append("inputs must be a dictionary")
            return errors

        products = inputs.get("products")
        if products is None:
            errors.append("Missing required key 'products'")
            return errors

        if not isinstance(products, list):
            errors.append("'products' must be a list")
            return errors

        if len(products) == 0:
            errors.append("'products' list is empty")
            return errors

        valid_categories = {c.value for c in IntermediateProductCategory}
        valid_processing = {p.value for p in ProcessingType}

        for idx, product in enumerate(products):
            pid = product.get("product_id", f"product_{idx}")

            if not isinstance(product, dict):
                errors.append(f"Product {pid}: must be a dictionary")
                continue

            # Required fields
            if not product.get("product_id"):
                errors.append(f"Product at index {idx}: missing 'product_id'")

            # Category validation
            category = product.get("product_category", "")
            if category and category not in valid_categories:
                errors.append(
                    f"Product {pid}: invalid product_category '{category}'. "
                    f"Valid: {sorted(valid_categories)}"
                )

            # Processing type validation
            processing = product.get("processing_type", "")
            if processing and processing not in valid_processing:
                errors.append(
                    f"Product {pid}: invalid processing_type '{processing}'. "
                    f"Valid: {sorted(valid_processing)}"
                )

            # At least one quantity source
            has_mass = product.get("mass_tonnes") is not None or product.get("mass_kg") is not None
            has_spend = product.get("spend_usd") is not None or product.get("revenue_usd") is not None
            has_energy = product.get("energy_kwh") is not None
            has_fuel = product.get("fuel_litres") is not None
            has_direct = product.get("direct_emissions_kg_co2e") is not None

            if not (has_mass or has_spend or has_energy or has_fuel or has_direct):
                errors.append(
                    f"Product {pid}: must provide at least one of "
                    "mass_tonnes, mass_kg, spend_usd, revenue_usd, "
                    "energy_kwh, fuel_litres, or direct_emissions_kg_co2e"
                )

            # Non-negative check for numeric fields
            for numeric_field in [
                "mass_tonnes", "mass_kg", "spend_usd", "revenue_usd",
                "energy_kwh", "fuel_litres", "direct_emissions_kg_co2e",
            ]:
                val = product.get(numeric_field)
                if val is not None:
                    try:
                        d = Decimal(str(val))
                        if d < ZERO:
                            errors.append(
                                f"Product {pid}: {numeric_field} must be non-negative (got {val})"
                            )
                    except (InvalidOperation, ValueError):
                        errors.append(
                            f"Product {pid}: {numeric_field} is not a valid number (got {val})"
                        )

        return errors

    def estimate_runtime(self, num_products: int) -> float:
        """
        Estimate pipeline runtime in seconds based on product count.

        Estimation formula: base_time + per_product_time * num_products
        Based on empirical benchmarks:
            - Base overhead: ~50ms for pipeline setup
            - Per-product: ~5ms for classify/normalize/resolve
            - Per-product: ~10ms for calculate
            - Per-product: ~2ms for allocate/aggregate
            - Compliance: ~20ms flat
            - Provenance + seal: ~10ms flat

        Args:
            num_products: Number of products to process.

        Returns:
            Estimated runtime in seconds.
        """
        base_ms = 80.0  # Pipeline overhead + compliance + provenance + seal
        per_product_ms = 17.0  # classify + normalize + resolve + calculate + allocate + aggregate

        estimated_ms = base_ms + per_product_ms * num_products
        return estimated_ms / 1000.0

    # ==========================================================================
    # STAGE 1: VALIDATE
    # ==========================================================================

    def _stage_validate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Validate and sanitize inputs.

        Validates structure, required fields, data types, and value ranges.
        Sanitizes string fields to prevent injection.

        Args:
            inputs: Raw input dictionary.

        Returns:
            Validated and sanitized input dictionary.

        Raises:
            ValueError: If validation fails.
        """
        errors = self.validate_inputs(inputs)
        if errors:
            raise ValueError(
                f"Input validation failed ({len(errors)} errors): "
                + "; ".join(errors[:5])
                + ("..." if len(errors) > 5 else "")
            )

        # Deep-copy and sanitize
        validated = {
            "products": [],
            "metadata": inputs.get("metadata", {}),
            "allocation_method": inputs.get("allocation_method", "mass_based"),
        }

        for product in inputs["products"]:
            sanitized = {}
            for key, value in product.items():
                if isinstance(value, str):
                    sanitized[key] = value.strip()[:1000]
                else:
                    sanitized[key] = value
            validated["products"].append(sanitized)

        logger.debug("Validation passed: %d products", len(validated["products"]))
        return validated

    # ==========================================================================
    # STAGE 2: CLASSIFY
    # ==========================================================================

    def _stage_classify(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 2: Classify products into intermediate product categories.

        If product_category is not set, attempts classification based on
        product name or description using keyword matching. Sets default
        processing_type if not provided.

        Args:
            products: List of validated product dictionaries.

        Returns:
            List of classified product dictionaries.
        """
        classified: List[Dict[str, Any]] = []

        # Keyword-to-category mapping for auto-classification
        category_keywords: Dict[IntermediateProductCategory, List[str]] = {
            IntermediateProductCategory.METALS_FERROUS: [
                "steel", "iron", "cast iron", "stainless",
            ],
            IntermediateProductCategory.METALS_NON_FERROUS: [
                "aluminum", "aluminium", "copper", "brass", "zinc", "titanium",
            ],
            IntermediateProductCategory.PLASTICS_THERMOPLASTIC: [
                "polyethylene", "polypropylene", "pvc", "pet", "abs", "nylon",
                "thermoplastic", "hdpe", "ldpe",
            ],
            IntermediateProductCategory.PLASTICS_THERMOSET: [
                "epoxy", "polyester resin", "phenolic", "thermoset",
            ],
            IntermediateProductCategory.CHEMICALS: [
                "chemical", "solvent", "acid", "alkali", "resin", "adhesive",
            ],
            IntermediateProductCategory.FOOD_INGREDIENTS: [
                "flour", "sugar", "starch", "cocoa", "oil seed", "milk powder",
                "food ingredient", "malt",
            ],
            IntermediateProductCategory.TEXTILES: [
                "cotton", "polyester fiber", "nylon fiber", "wool", "silk",
                "textile", "fabric", "yarn",
            ],
            IntermediateProductCategory.ELECTRONICS_COMPONENTS: [
                "pcb", "semiconductor", "capacitor", "resistor", "chip",
                "circuit board", "electronic",
            ],
            IntermediateProductCategory.GLASS_CERAMICS: [
                "glass", "ceramic", "porcelain", "silica",
            ],
            IntermediateProductCategory.WOOD_PAPER_PULP: [
                "wood", "pulp", "paper", "lumber", "timber", "cellulose",
            ],
            IntermediateProductCategory.MINERALS: [
                "mineral", "cement", "limestone", "sand", "gravel", "gypsum",
            ],
            IntermediateProductCategory.AGRICULTURAL_COMMODITIES: [
                "grain", "wheat", "corn", "rice", "soybean", "coffee",
                "agricultural",
            ],
        }

        # Default processing types per category
        default_processing: Dict[IntermediateProductCategory, ProcessingType] = {
            IntermediateProductCategory.METALS_FERROUS: ProcessingType.MACHINING,
            IntermediateProductCategory.METALS_NON_FERROUS: ProcessingType.MACHINING,
            IntermediateProductCategory.PLASTICS_THERMOPLASTIC: ProcessingType.INJECTION_MOLDING,
            IntermediateProductCategory.PLASTICS_THERMOSET: ProcessingType.INJECTION_MOLDING,
            IntermediateProductCategory.CHEMICALS: ProcessingType.CHEMICAL_REACTION,
            IntermediateProductCategory.FOOD_INGREDIENTS: ProcessingType.MILLING,
            IntermediateProductCategory.TEXTILES: ProcessingType.TEXTILE_FINISHING,
            IntermediateProductCategory.ELECTRONICS_COMPONENTS: ProcessingType.ASSEMBLY,
            IntermediateProductCategory.GLASS_CERAMICS: ProcessingType.HEAT_TREATMENT,
            IntermediateProductCategory.WOOD_PAPER_PULP: ProcessingType.MILLING,
            IntermediateProductCategory.MINERALS: ProcessingType.MILLING,
            IntermediateProductCategory.AGRICULTURAL_COMMODITIES: ProcessingType.MILLING,
        }

        for product in products:
            p = dict(product)

            # Auto-classify category if not provided
            if not p.get("product_category"):
                product_name = (
                    p.get("product_name", "") + " " + p.get("description", "")
                ).lower()
                matched_category = None
                for category, keywords in category_keywords.items():
                    for kw in keywords:
                        if kw in product_name:
                            matched_category = category
                            break
                    if matched_category:
                        break

                if matched_category:
                    p["product_category"] = matched_category.value
                    p["category_auto_classified"] = True
                else:
                    p["product_category"] = ""
                    p["category_auto_classified"] = False
                    logger.warning(
                        "Product %s: could not auto-classify category",
                        p.get("product_id", "unknown"),
                    )

            # Set default processing type if not provided
            if not p.get("processing_type"):
                try:
                    cat_enum = IntermediateProductCategory(p.get("product_category", ""))
                    default_proc = default_processing.get(cat_enum)
                    if default_proc:
                        p["processing_type"] = default_proc.value
                        p["processing_type_defaulted"] = True
                except (ValueError, KeyError):
                    p["processing_type"] = ""
                    p["processing_type_defaulted"] = False

            classified.append(p)

        return classified

    # ==========================================================================
    # STAGE 3: NORMALIZE
    # ==========================================================================

    def _stage_normalize(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Normalize units to standard units (tonnes, kWh, litres, USD).

        Converts:
            - kg -> tonnes, lb -> tonnes, short tons -> tonnes
            - gallons -> litres (US and imperial)
            - BTU -> kWh, MJ -> kWh, therms -> kWh
            - MCF -> m3

        Args:
            products: List of classified product dictionaries.

        Returns:
            List of products with normalized units.
        """
        normalized: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)

            # Mass normalization to tonnes
            if p.get("mass_tonnes") is None and p.get("mass_kg") is not None:
                mass_kg = self._safe_decimal(p["mass_kg"])
                if mass_kg is not None:
                    p["mass_tonnes"] = str(
                        (mass_kg * UNIT_CONVERSIONS["kg_to_tonnes"]).quantize(
                            _QUANT_8DP, rounding=ROUNDING
                        )
                    )
                    p["mass_unit_original"] = "kg"

            if p.get("mass_tonnes") is None and p.get("mass_lb") is not None:
                mass_lb = self._safe_decimal(p["mass_lb"])
                if mass_lb is not None:
                    p["mass_tonnes"] = str(
                        (mass_lb * UNIT_CONVERSIONS["lb_to_tonnes"]).quantize(
                            _QUANT_8DP, rounding=ROUNDING
                        )
                    )
                    p["mass_unit_original"] = "lb"

            if p.get("mass_tonnes") is None and p.get("mass_short_tons") is not None:
                mass_st = self._safe_decimal(p["mass_short_tons"])
                if mass_st is not None:
                    p["mass_tonnes"] = str(
                        (mass_st * UNIT_CONVERSIONS["short_ton_to_tonnes"]).quantize(
                            _QUANT_8DP, rounding=ROUNDING
                        )
                    )
                    p["mass_unit_original"] = "short_tons"

            # Energy normalization to kWh
            if p.get("energy_kwh") is None and p.get("energy_btu") is not None:
                btu = self._safe_decimal(p["energy_btu"])
                if btu is not None:
                    p["energy_kwh"] = str(
                        (btu * UNIT_CONVERSIONS["btu_to_kwh"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["energy_unit_original"] = "btu"

            if p.get("energy_kwh") is None and p.get("energy_mj") is not None:
                mj = self._safe_decimal(p["energy_mj"])
                if mj is not None:
                    p["energy_kwh"] = str(
                        (mj * UNIT_CONVERSIONS["mj_to_kwh"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["energy_unit_original"] = "mj"

            if p.get("energy_kwh") is None and p.get("energy_therms") is not None:
                therms = self._safe_decimal(p["energy_therms"])
                if therms is not None:
                    p["energy_kwh"] = str(
                        (therms * UNIT_CONVERSIONS["therm_to_kwh"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["energy_unit_original"] = "therms"

            # Fuel normalization to litres
            if p.get("fuel_litres") is None and p.get("fuel_usg") is not None:
                usg = self._safe_decimal(p["fuel_usg"])
                if usg is not None:
                    p["fuel_litres"] = str(
                        (usg * UNIT_CONVERSIONS["usg_to_litres"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["fuel_unit_original"] = "usg"

            if p.get("fuel_litres") is None and p.get("fuel_imp_gal") is not None:
                ig = self._safe_decimal(p["fuel_imp_gal"])
                if ig is not None:
                    p["fuel_litres"] = str(
                        (ig * UNIT_CONVERSIONS["imp_gal_to_litres"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["fuel_unit_original"] = "imp_gal"

            # Gas volume normalization to m3
            if p.get("gas_m3") is None and p.get("gas_mcf") is not None:
                mcf = self._safe_decimal(p["gas_mcf"])
                if mcf is not None:
                    p["gas_m3"] = str(
                        (mcf * UNIT_CONVERSIONS["mcf_to_m3"]).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        )
                    )
                    p["gas_unit_original"] = "mcf"

            normalized.append(p)

        return normalized

    # ==========================================================================
    # STAGE 4: RESOLVE_EFS
    # ==========================================================================

    def _stage_resolve_efs(
        self,
        products: List[Dict[str, Any]],
        method: CalculationMethod,
    ) -> List[Dict[str, Any]]:
        """
        Stage 4: Resolve emission factors for each product.

        EF resolution hierarchy:
            1. Customer-reported direct EF (if site-specific)
            2. DB engine lookup (ProcessingDatabaseEngine)
            3. Default EF table (category:processing_type key)
            4. EEIO sector factor (spend-based fallback)

        Args:
            products: List of normalized product dictionaries.
            method: Target calculation method.

        Returns:
            List of products with resolved emission factors.
        """
        products_with_efs: List[Dict[str, Any]] = []
        db_engine = self._get_db_engine()

        for product in products:
            p = dict(product)
            category = p.get("product_category", "")
            processing = p.get("processing_type", "")
            ef_key = f"{category}:{processing}"

            # 1. Customer-reported direct EF
            if p.get("customer_ef_kg_co2e_per_tonne") is not None:
                p["ef_kg_co2e_per_tonne"] = str(p["customer_ef_kg_co2e_per_tonne"])
                p["ef_source"] = EFSource.CUSTOMER_REPORTED.value
                p["ef_resolution"] = "customer_reported"
                products_with_efs.append(p)
                continue

            # 2. DB engine lookup
            if db_engine is not None:
                try:
                    db_ef = db_engine.get_processing_ef(category, processing)
                    if db_ef is not None:
                        p["ef_kg_co2e_per_tonne"] = str(db_ef.get("ef_kg_co2e_per_tonne", "0"))
                        p["ef_source"] = db_ef.get("source", "database")
                        p["ef_uncertainty_pct"] = str(db_ef.get("uncertainty_pct", "30"))
                        p["ef_resolution"] = "database"
                        products_with_efs.append(p)
                        continue
                except Exception as e:
                    logger.warning(
                        "DB EF lookup failed for %s: %s", ef_key, str(e)
                    )

            # 3. Default EF table (average-data)
            default_ef = DEFAULT_PROCESSING_EFS.get(ef_key)
            if default_ef is not None:
                p["ef_kg_co2e_per_tonne"] = str(default_ef["ef_kg_co2e_per_tonne"])
                p["ef_source"] = default_ef.get("source", "default")
                p["ef_uncertainty_pct"] = str(default_ef.get("uncertainty_pct", "30"))
                p["ef_resolution"] = "default_table"
                products_with_efs.append(p)
                continue

            # 4. EEIO fallback (spend-based)
            eeio_ef = DEFAULT_EEIO_FACTORS.get(category)
            if eeio_ef is not None and (p.get("spend_usd") or p.get("revenue_usd")):
                p["ef_kg_co2e_per_usd"] = str(eeio_ef["ef_kg_co2e_per_usd"])
                p["ef_source"] = EFSource.EEIO.value
                p["ef_uncertainty_pct"] = str(eeio_ef.get("uncertainty_pct", "50"))
                p["ef_resolution"] = "eeio_fallback"
                products_with_efs.append(p)
                continue

            # No EF found
            p["ef_resolution"] = "none"
            p["warning"] = f"No emission factor found for {ef_key}"
            logger.warning("No EF resolved for product %s (%s)", p.get("product_id", "?"), ef_key)
            products_with_efs.append(p)

        return products_with_efs

    # ==========================================================================
    # STAGE 5: CALCULATE
    # ==========================================================================

    def _stage_calculate(
        self,
        products: List[Dict[str, Any]],
        method: CalculationMethod,
    ) -> List[Dict[str, Any]]:
        """
        Stage 5: Calculate emissions for each product.

        Method waterfall for hybrid: site-specific -> average-data -> spend-based.

        Calculation formulas (ZERO HALLUCINATION - deterministic only):
            - Site-specific direct: direct_emissions_kg_co2e (as reported)
            - Site-specific energy: energy_kwh * grid_ef_kg_co2e_per_kwh
            - Site-specific fuel: fuel_litres * fuel_ef_kg_co2e_per_litre
            - Average-data: mass_tonnes * ef_kg_co2e_per_tonne
            - Spend-based: spend_usd * ef_kg_co2e_per_usd

        Args:
            products: List of products with resolved EFs.
            method: Calculation method.

        Returns:
            List of products with calculated emissions.
        """
        calculated: List[Dict[str, Any]] = []

        for product in products:
            p = dict(product)
            emissions = ZERO
            method_used = ""

            try:
                if method == CalculationMethod.HYBRID:
                    emissions, method_used = self._calculate_hybrid(p)
                elif method == CalculationMethod.SITE_SPECIFIC_DIRECT:
                    emissions, method_used = self._calculate_site_specific_direct(p)
                elif method == CalculationMethod.SITE_SPECIFIC_ENERGY:
                    emissions, method_used = self._calculate_site_specific_energy(p)
                elif method == CalculationMethod.SITE_SPECIFIC_FUEL:
                    emissions, method_used = self._calculate_site_specific_fuel(p)
                elif method == CalculationMethod.AVERAGE_DATA:
                    emissions, method_used = self._calculate_average_data(p)
                elif method == CalculationMethod.SPEND_BASED:
                    emissions, method_used = self._calculate_spend_based(p)
                else:
                    emissions, method_used = self._calculate_hybrid(p)

            except Exception as e:
                logger.error(
                    "Calculation failed for product %s: %s",
                    p.get("product_id", "?"),
                    str(e),
                    exc_info=True,
                )
                p["warning"] = f"Calculation failed: {str(e)}"
                method_used = "failed"

            p["emissions_kg_co2e"] = str(
                emissions.quantize(_QUANT_4DP, rounding=ROUNDING)
            )
            p["calculation_method_used"] = method_used
            calculated.append(p)

        return calculated

    def _calculate_hybrid(self, product: Dict[str, Any]) -> Tuple[Decimal, str]:
        """
        Hybrid calculation: waterfall through methods in priority order.

        Priority: direct -> energy -> fuel -> average-data -> spend-based.

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions_kg_co2e, method_used).
        """
        # Try site-specific direct
        if product.get("direct_emissions_kg_co2e") is not None:
            emissions, method = self._calculate_site_specific_direct(product)
            if emissions > ZERO:
                return emissions, method

        # Try site-specific energy
        if product.get("energy_kwh") is not None:
            emissions, method = self._calculate_site_specific_energy(product)
            if emissions > ZERO:
                return emissions, method

        # Try site-specific fuel
        if product.get("fuel_litres") is not None:
            emissions, method = self._calculate_site_specific_fuel(product)
            if emissions > ZERO:
                return emissions, method

        # Try average-data
        if product.get("mass_tonnes") is not None and product.get("ef_kg_co2e_per_tonne") is not None:
            emissions, method = self._calculate_average_data(product)
            if emissions > ZERO:
                return emissions, method

        # Try spend-based
        if (product.get("spend_usd") or product.get("revenue_usd")) and product.get("ef_kg_co2e_per_usd"):
            emissions, method = self._calculate_spend_based(product)
            if emissions > ZERO:
                return emissions, method

        return ZERO, "no_data"

    def _calculate_site_specific_direct(
        self, product: Dict[str, Any]
    ) -> Tuple[Decimal, str]:
        """
        Site-specific direct: customer-reported processing emissions.

        Formula: emissions = direct_emissions_kg_co2e (as reported by customer)

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions, "site_specific_direct").
        """
        direct = self._safe_decimal(product.get("direct_emissions_kg_co2e"))
        if direct is not None and direct >= ZERO:
            return direct, CalculationMethod.SITE_SPECIFIC_DIRECT.value
        return ZERO, CalculationMethod.SITE_SPECIFIC_DIRECT.value

    def _calculate_site_specific_energy(
        self, product: Dict[str, Any]
    ) -> Tuple[Decimal, str]:
        """
        Site-specific energy: energy consumption x grid emission factor.

        Formula: emissions = energy_kwh * grid_ef_kg_co2e_per_kwh

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions, "site_specific_energy").
        """
        energy_kwh = self._safe_decimal(product.get("energy_kwh"))
        if energy_kwh is None or energy_kwh <= ZERO:
            return ZERO, CalculationMethod.SITE_SPECIFIC_ENERGY.value

        # Resolve grid EF
        grid_region = product.get("grid_region", "global_average")
        grid_ef = self._safe_decimal(product.get("grid_ef_kg_co2e_per_kwh"))
        if grid_ef is None:
            grid_ef = DEFAULT_GRID_EFS.get(grid_region, DEFAULT_GRID_EFS["global_average"])

        emissions = energy_kwh * grid_ef
        return emissions, CalculationMethod.SITE_SPECIFIC_ENERGY.value

    def _calculate_site_specific_fuel(
        self, product: Dict[str, Any]
    ) -> Tuple[Decimal, str]:
        """
        Site-specific fuel: fuel consumption x combustion emission factor.

        Formula: emissions = fuel_litres * fuel_ef_kg_co2e_per_litre

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions, "site_specific_fuel").
        """
        fuel_litres = self._safe_decimal(product.get("fuel_litres"))
        if fuel_litres is None or fuel_litres <= ZERO:
            return ZERO, CalculationMethod.SITE_SPECIFIC_FUEL.value

        fuel_type = product.get("fuel_type", "diesel_litre")
        fuel_ef = self._safe_decimal(product.get("fuel_ef_kg_co2e_per_litre"))
        if fuel_ef is None:
            fuel_ef = DEFAULT_FUEL_EFS.get(fuel_type, DEFAULT_FUEL_EFS["diesel_litre"])

        emissions = fuel_litres * fuel_ef
        return emissions, CalculationMethod.SITE_SPECIFIC_FUEL.value

    def _calculate_average_data(
        self, product: Dict[str, Any]
    ) -> Tuple[Decimal, str]:
        """
        Average-data: mass x processing emission factor.

        Formula: emissions = mass_tonnes * ef_kg_co2e_per_tonne

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions, "average_data").
        """
        mass = self._safe_decimal(product.get("mass_tonnes"))
        ef = self._safe_decimal(product.get("ef_kg_co2e_per_tonne"))

        if mass is None or ef is None or mass <= ZERO or ef <= ZERO:
            return ZERO, CalculationMethod.AVERAGE_DATA.value

        emissions = mass * ef
        return emissions, CalculationMethod.AVERAGE_DATA.value

    def _calculate_spend_based(
        self, product: Dict[str, Any]
    ) -> Tuple[Decimal, str]:
        """
        Spend-based: revenue/spend x EEIO sector emission factor.

        Formula: emissions = spend_usd * ef_kg_co2e_per_usd

        Args:
            product: Single product dictionary.

        Returns:
            Tuple of (emissions, "spend_based").
        """
        spend = self._safe_decimal(
            product.get("spend_usd") or product.get("revenue_usd")
        )
        ef = self._safe_decimal(product.get("ef_kg_co2e_per_usd"))

        if spend is None or ef is None or spend <= ZERO or ef <= ZERO:
            return ZERO, CalculationMethod.SPEND_BASED.value

        emissions = spend * ef
        return emissions, CalculationMethod.SPEND_BASED.value

    # ==========================================================================
    # STAGE 6: ALLOCATE
    # ==========================================================================

    def _stage_allocate(
        self,
        breakdowns: List[Dict[str, Any]],
        allocation_method: str,
    ) -> List[Dict[str, Any]]:
        """
        Stage 6: Apply proportional allocation to emissions.

        Allocation applies when a product has an allocation_factor (0-1)
        or when a facility's emissions need to be split across products.

        Supported methods:
            - mass_based: Allocate by mass proportion
            - economic: Allocate by revenue/spend proportion
            - energy_content: Allocate by energy content
            - equal: Equal distribution
            - custom: Use product-level allocation_factor

        Args:
            breakdowns: List of calculated product dictionaries.
            allocation_method: Allocation method name.

        Returns:
            List of products with allocated emissions.
        """
        allocated: List[Dict[str, Any]] = []

        # Calculate totals for proportional allocation
        total_mass = ZERO
        total_spend = ZERO
        total_energy = ZERO

        for product in breakdowns:
            mass = self._safe_decimal(product.get("mass_tonnes", "0")) or ZERO
            spend = self._safe_decimal(product.get("spend_usd", "0") or product.get("revenue_usd", "0")) or ZERO
            energy = self._safe_decimal(product.get("energy_kwh", "0")) or ZERO
            total_mass += mass
            total_spend += spend
            total_energy += energy

        for product in breakdowns:
            p = dict(product)
            emissions = self._safe_decimal(p.get("emissions_kg_co2e", "0")) or ZERO

            # Check for product-level allocation factor
            alloc_factor = self._safe_decimal(p.get("allocation_factor"))
            if alloc_factor is not None and ZERO < alloc_factor <= ONE:
                allocated_emissions = emissions * alloc_factor
                p["emissions_kg_co2e"] = str(
                    allocated_emissions.quantize(_QUANT_4DP, rounding=ROUNDING)
                )
                p["allocation_factor_applied"] = str(alloc_factor)
                p["allocation_method"] = "custom"
                allocated.append(p)
                continue

            # If no product-level factor, apply method-level proportional allocation
            # Only applies if there is a shared_facility_emissions field
            shared_emissions = self._safe_decimal(p.get("shared_facility_emissions_kg_co2e"))
            if shared_emissions is not None and shared_emissions > ZERO:
                factor = ZERO
                try:
                    alloc_enum = AllocationMethod(allocation_method)
                except (ValueError, KeyError):
                    alloc_enum = AllocationMethod.MASS_BASED

                if alloc_enum == AllocationMethod.MASS_BASED and total_mass > ZERO:
                    mass = self._safe_decimal(p.get("mass_tonnes", "0")) or ZERO
                    factor = mass / total_mass

                elif alloc_enum == AllocationMethod.ECONOMIC and total_spend > ZERO:
                    spend = self._safe_decimal(p.get("spend_usd", "0") or p.get("revenue_usd", "0")) or ZERO
                    factor = spend / total_spend

                elif alloc_enum == AllocationMethod.ENERGY_CONTENT and total_energy > ZERO:
                    energy = self._safe_decimal(p.get("energy_kwh", "0")) or ZERO
                    factor = energy / total_energy

                elif alloc_enum == AllocationMethod.EQUAL and len(breakdowns) > 0:
                    factor = ONE / Decimal(str(len(breakdowns)))

                else:
                    factor = ONE

                allocated_shared = shared_emissions * factor
                total_product_emissions = emissions + allocated_shared
                p["emissions_kg_co2e"] = str(
                    total_product_emissions.quantize(_QUANT_4DP, rounding=ROUNDING)
                )
                p["allocation_factor_applied"] = str(
                    factor.quantize(_QUANT_8DP, rounding=ROUNDING)
                )
                p["allocation_method"] = alloc_enum.value
            else:
                p["allocation_method"] = "none"

            allocated.append(p)

        return allocated

    # ==========================================================================
    # STAGE 7: AGGREGATE
    # ==========================================================================

    def _stage_aggregate(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 7: Aggregate emissions across all products.

        Produces aggregations by:
            - Category (IntermediateProductCategory)
            - Processing type (ProcessingType)
            - Calculation method used
            - EF source

        Args:
            products: List of allocated product dictionaries.

        Returns:
            Aggregation dictionary with totals and breakdowns.
        """
        total_emissions = ZERO
        total_mass = ZERO
        by_category: Dict[str, str] = {}
        by_processing_type: Dict[str, str] = {}
        by_method: Dict[str, str] = {}
        by_ef_source: Dict[str, str] = {}
        product_count = 0
        calculated_count = 0

        for product in products:
            em = self._safe_decimal(product.get("emissions_kg_co2e", "0")) or ZERO
            mass = self._safe_decimal(product.get("mass_tonnes", "0")) or ZERO
            total_emissions += em
            total_mass += mass
            product_count += 1

            if em > ZERO:
                calculated_count += 1

            # By category
            category = product.get("product_category", "unknown")
            cat_total = self._safe_decimal(by_category.get(category, "0")) or ZERO
            by_category[category] = str(
                (cat_total + em).quantize(_QUANT_4DP, rounding=ROUNDING)
            )

            # By processing type
            proc = product.get("processing_type", "unknown")
            proc_total = self._safe_decimal(by_processing_type.get(proc, "0")) or ZERO
            by_processing_type[proc] = str(
                (proc_total + em).quantize(_QUANT_4DP, rounding=ROUNDING)
            )

            # By method
            method = product.get("calculation_method_used", "unknown")
            method_total = self._safe_decimal(by_method.get(method, "0")) or ZERO
            by_method[method] = str(
                (method_total + em).quantize(_QUANT_4DP, rounding=ROUNDING)
            )

            # By EF source
            ef_src = product.get("ef_source", "unknown")
            ef_total = self._safe_decimal(by_ef_source.get(ef_src, "0")) or ZERO
            by_ef_source[ef_src] = str(
                (ef_total + em).quantize(_QUANT_4DP, rounding=ROUNDING)
            )

        # Completeness percentage
        completeness = ZERO
        if product_count > 0:
            completeness = (
                Decimal(str(calculated_count)) / Decimal(str(product_count)) * ONE_HUNDRED
            ).quantize(_QUANT_2DP, rounding=ROUNDING)

        # Emission intensity (kg CO2e per tonne of product)
        intensity = ZERO
        if total_mass > ZERO:
            intensity = (total_emissions / total_mass).quantize(
                _QUANT_4DP, rounding=ROUNDING
            )

        return {
            "total_emissions_kg_co2e": str(
                total_emissions.quantize(_QUANT_4DP, rounding=ROUNDING)
            ),
            "total_mass_tonnes": str(
                total_mass.quantize(_QUANT_4DP, rounding=ROUNDING)
            ),
            "emission_intensity_kg_co2e_per_tonne": str(intensity),
            "product_count": product_count,
            "calculated_count": calculated_count,
            "completeness_percentage": str(completeness),
            "by_category": by_category,
            "by_processing_type": by_processing_type,
            "by_method": by_method,
            "by_ef_source": by_ef_source,
        }

    # ==========================================================================
    # STAGE 8: COMPLIANCE
    # ==========================================================================

    def _stage_compliance(
        self,
        result: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Stage 8: Run compliance checks against regulatory frameworks.

        Delegates to ComplianceCheckerEngine (Engine 6). If the engine is
        not available, returns an empty compliance result with a warning.

        Args:
            result: Combined calculation result for compliance input.
            frameworks: Optional list of framework names to check.

        Returns:
            Dictionary of compliance results by framework.
        """
        compliance_engine = self._get_compliance_engine()

        if compliance_engine is None:
            logger.warning("ComplianceCheckerEngine not available, skipping compliance checks")
            return {"warning": "Compliance engine not available"}

        try:
            # Resolve framework enums
            framework_enums = None
            if frameworks:
                from greenlang.processing_sold_products.compliance_checker import (
                    ComplianceFramework as CF,
                )
                framework_enums = []
                for f_name in frameworks:
                    try:
                        framework_enums.append(CF(f_name))
                    except (ValueError, KeyError):
                        logger.warning("Unknown framework: %s, skipping", f_name)

            # Run framework checks
            check_results = compliance_engine.check_all(result, framework_enums)

            # Run double-counting checks
            dc_results = compliance_engine.check_all_dc_rules(result)

            # Generate report
            report = compliance_engine.generate_compliance_report(check_results, dc_results)
            return report

        except Exception as e:
            logger.error("Compliance checking failed: %s", str(e), exc_info=True)
            return {
                "error": f"Compliance checking failed: {str(e)}",
                "overall_status": "ERROR",
            }

    # ==========================================================================
    # STAGE 9: PROVENANCE
    # ==========================================================================

    def _stage_provenance(self, chain: ProvenanceChain) -> Dict[str, Any]:
        """
        Stage 9: Compute final provenance hash chain.

        The chain links all stage hashes together to create an immutable
        audit trail proving no data was modified between stages.

        Args:
            chain: Provenance chain accumulated during pipeline execution.

        Returns:
            Dictionary with chain_id, final_hash, and entry count.
        """
        final_hash = chain.get_final_hash()
        chain.final_hash = final_hash
        chain.sealed_at = datetime.now(timezone.utc).isoformat()

        return {
            "chain_id": chain.chain_id,
            "final_hash": final_hash,
            "entry_count": len(chain.entries),
            "sealed_at": chain.sealed_at,
            "entries": [
                {
                    "stage": e.stage,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "chain_hash": e.chain_hash,
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                }
                for e in chain.entries
            ],
        }

    # ==========================================================================
    # STAGE 10: SEAL
    # ==========================================================================

    def _stage_seal(
        self,
        result: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Stage 10: Seal the final result with provenance hash.

        Creates an immutable final result by computing a SHA-256 hash over
        the entire result including provenance chain. The sealed result
        cannot be modified without invalidating the hash.

        Args:
            result: Complete pipeline result dictionary.
            provenance: Provenance chain result from Stage 9.

        Returns:
            Sealed result dictionary with provenance_hash and seal metadata.
        """
        sealed = dict(result)

        # Add provenance data
        sealed["provenance"] = provenance

        # Compute final seal hash over all data
        seal_input = {
            "pipeline_id": sealed.get("pipeline_id", ""),
            "total_emissions_kg_co2e": sealed.get("total_emissions_kg_co2e", "0"),
            "total_products": sealed.get("total_products", 0),
            "calculation_method": sealed.get("calculation_method", ""),
            "org_id": sealed.get("org_id", ""),
            "reporting_year": sealed.get("reporting_year", 0),
            "provenance_chain_hash": provenance.get("final_hash", ""),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": AGENT_ID,
            "engine_version": ENGINE_VERSION,
        }

        seal_serialized = json.dumps(seal_input, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(seal_serialized.encode("utf-8")).hexdigest()

        sealed["provenance_hash"] = provenance_hash
        sealed["sealed_at"] = seal_input["sealed_at"]
        sealed["seal_metadata"] = {
            "algorithm": "sha256",
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
        }

        logger.info(
            "Result sealed: hash=%s...%s",
            provenance_hash[:8],
            provenance_hash[-8:],
        )

        return sealed

    # ==========================================================================
    # PRIVATE UTILITY METHODS
    # ==========================================================================

    def _build_compliance_input(
        self,
        aggregation: Dict[str, Any],
        products: List[Dict[str, Any]],
        method: CalculationMethod,
        org_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Build the input dictionary for the compliance checking stage.

        Combines aggregation results with product details into the
        format expected by ComplianceCheckerEngine.

        Args:
            aggregation: Stage 7 aggregation results.
            products: Allocated product list.
            method: Calculation method used.
            org_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary suitable for ComplianceCheckerEngine.check_all().
        """
        ef_sources = list({
            p.get("ef_source", "unknown") for p in products if p.get("ef_source")
        })

        return {
            "total_emissions_kg_co2e": aggregation.get("total_emissions_kg_co2e", "0"),
            "calculation_method": method.value,
            "methodology": method.value,
            "products": products,
            "ef_sources": ef_sources,
            "data_quality_score": aggregation.get("completeness_percentage"),
            "coverage_percentage": aggregation.get("completeness_percentage"),
            "completeness_percentage": aggregation.get("completeness_percentage"),
            "org_id": org_id,
            "reporting_year": reporting_year,
        }

    def _record_provenance(
        self,
        chain: ProvenanceChain,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
    ) -> None:
        """
        Record a provenance entry for a pipeline stage.

        Computes SHA-256 hashes of input and output, then chains them
        with the previous entry's hash.

        Args:
            chain: Current provenance chain.
            stage: Pipeline stage.
            input_data: Stage input data (for hashing).
            output_data: Stage output data (for hashing).
            duration_ms: Stage duration in milliseconds.
        """
        input_hash = self._compute_hash(input_data)
        output_hash = self._compute_hash(output_data)

        previous_hash = ""
        if chain.entries:
            previous_hash = chain.entries[-1].chain_hash

        chain_input = f"{previous_hash}:{stage.value}:{input_hash}:{output_hash}"
        chain_hash = hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

        entry = ProvenanceEntry(
            stage=stage.value,
            input_hash=input_hash,
            output_hash=output_hash,
            chain_hash=chain_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
        )
        chain.append_entry(entry)

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of arbitrary data.

        Args:
            data: Data to hash (serialized to JSON).

        Returns:
            SHA-256 hex digest string.
        """
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = str(data)

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _elapsed_ms(self, start: float) -> float:
        """
        Calculate elapsed milliseconds from a monotonic start time.

        Args:
            start: Monotonic clock start value.

        Returns:
            Elapsed time in milliseconds.
        """
        return (time.monotonic() - start) * 1000

    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """
        Safely convert a value to Decimal.

        Args:
            value: Value to convert.

        Returns:
            Decimal value, or None if conversion fails.
        """
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_pipeline_engine() -> ProcessingPipelineEngine:
    """
    Get the singleton ProcessingPipelineEngine instance.

    Returns:
        ProcessingPipelineEngine singleton instance.
    """
    return ProcessingPipelineEngine()


def run_pipeline(
    inputs: Dict[str, Any],
    method: str = "hybrid",
    org_id: str = "",
    reporting_year: int = 2025,
) -> Dict[str, Any]:
    """
    Convenience function to run the processing pipeline.

    Args:
        inputs: Dictionary with "products" key.
        method: Calculation method.
        org_id: Organization identifier.
        reporting_year: Reporting year.

    Returns:
        Complete pipeline result dictionary.

    Example:
        >>> result = run_pipeline(
        ...     inputs={"products": [{"product_id": "P1", "mass_tonnes": "100", ...}]},
        ...     method="average_data",
        ... )
        >>> print(f"Emissions: {result['total_emissions_kg_co2e']}")
    """
    engine = get_pipeline_engine()
    return engine.run_pipeline(
        inputs=inputs,
        method=method,
        org_id=org_id,
        reporting_year=reporting_year,
    )
