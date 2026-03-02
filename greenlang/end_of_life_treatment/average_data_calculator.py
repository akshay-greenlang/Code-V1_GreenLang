# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine - AGENT-MRV-025 Engine 3

GHG Protocol Scope 3 Category 12 average-data emissions calculator (Method B).

This engine calculates end-of-life treatment emissions for sold products using
pre-mixed composite emission factors by product category. It implements the
average-data approach, which is appropriate when detailed material composition
or producer-specific (EPD/PCF) data is unavailable.

Core Formula:
    E_total = SUM_products [ units_sold x weight_per_unit x average_eol_EF(product_type) ]

Each composite EF already blends typical regional treatment scenarios
(landfill %, incineration %, recycling %, composting %) into a single
kgCO2e/kg factor per product category. Regional adjustment factors are
then applied to account for differences in waste infrastructure (e.g., EU
has higher recycling than US).

Data Quality:
    Average-data is a Tier 2 method. The GHG Protocol recommends using
    producer-specific (Tier 1) or waste-type-specific data when available.
    Uncertainty is +/-30-50% for average-data estimates.

Thread Safety:
    Thread-safe singleton with threading.RLock() and double-checked locking.

References:
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 12
    - GHG Protocol Product Life Cycle Standard (2011)
    - IPCC 2006 Guidelines for National GHG Inventories, Vol 5
    - EPA WARM v16 (Waste Reduction Model)
    - DEFRA/DESNZ GHG Reporting Conversion Factors

Example:
    >>> engine = AverageDataCalculatorEngine.get_instance()
    >>> result = engine.calculate(
    ...     products=[{"product_id": "P-001", "category": "electronics",
    ...                "units_sold": 1000, "weight_per_unit_kg": Decimal("0.5")}],
    ...     org_id="ORG-001",
    ...     year=2025,
    ... )
    >>> result["total_co2e_kg"] > Decimal("0")
    True

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
"""

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# GRACEFUL IMPORTS
# ==============================================================================

try:
    from greenlang.end_of_life_treatment.config import get_config
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Fallback configuration stub."""
        return None

try:
    from greenlang.end_of_life_treatment.metrics import get_metrics
except ImportError:
    def get_metrics() -> Any:  # type: ignore[misc]
        """Fallback metrics stub."""
        return None

try:
    from greenlang.end_of_life_treatment.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment,misc]

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
ENGINE_ID: str = "average_data_calculator_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"

# ==============================================================================
# DECIMAL CONSTANTS
# ==============================================================================

PRECISION: int = 6
ROUNDING: str = ROUND_HALF_UP
_QUANT_6DP: Decimal = Decimal("0.000001")
_QUANT_2DP: Decimal = Decimal("0.01")
_ZERO: Decimal = Decimal("0")
_ONE: Decimal = Decimal("1")
KG_PER_TONNE: Decimal = Decimal("1000")
TONNES_PER_KG: Decimal = Decimal("0.001")

# ==============================================================================
# DATA TABLE: COMPOSITE EOL EMISSION FACTORS BY PRODUCT CATEGORY
# ==============================================================================
# Each factor is a pre-mixed kgCO2e/kg blending typical regional treatment
# scenarios (landfill / incineration / recycling / composting / open burning).
# Sources: EPA WARM v16, DEFRA/DESNZ 2025, IPCC 2006 Vol 5, ecoinvent 3.9.

COMPOSITE_EOL_EF: Dict[str, Dict[str, Any]] = {
    "electronics": {
        "ef_kgco2e_per_kg": Decimal("0.85"),
        "description": "Consumer electronics (phones, tablets, laptops, peripherals)",
        "treatment_mix": {
            "landfill": Decimal("0.25"),
            "incineration": Decimal("0.15"),
            "recycling": Decimal("0.55"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "appliances": {
        "ef_kgco2e_per_kg": Decimal("0.65"),
        "description": "Large and small household appliances (fridges, washers, microwaves)",
        "treatment_mix": {
            "landfill": Decimal("0.20"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.65"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "furniture": {
        "ef_kgco2e_per_kg": Decimal("0.45"),
        "description": "Home and office furniture (wood, metal, upholstered)",
        "treatment_mix": {
            "landfill": Decimal("0.45"),
            "incineration": Decimal("0.15"),
            "recycling": Decimal("0.30"),
            "composting": Decimal("0.05"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "packaging": {
        "ef_kgco2e_per_kg": Decimal("0.35"),
        "description": "Product packaging (cardboard, plastic, foam, paper)",
        "treatment_mix": {
            "landfill": Decimal("0.30"),
            "incineration": Decimal("0.15"),
            "recycling": Decimal("0.45"),
            "composting": Decimal("0.05"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "clothing": {
        "ef_kgco2e_per_kg": Decimal("0.55"),
        "description": "Clothing and textiles (natural and synthetic fibres)",
        "treatment_mix": {
            "landfill": Decimal("0.50"),
            "incineration": Decimal("0.20"),
            "recycling": Decimal("0.15"),
            "composting": Decimal("0.05"),
            "open_burning": Decimal("0.10"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "automotive_parts": {
        "ef_kgco2e_per_kg": Decimal("0.70"),
        "description": "Automotive components and spare parts (metal, plastic, rubber)",
        "treatment_mix": {
            "landfill": Decimal("0.20"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.65"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "building_materials": {
        "ef_kgco2e_per_kg": Decimal("0.40"),
        "description": "Construction and building materials (concrete, steel, insulation)",
        "treatment_mix": {
            "landfill": Decimal("0.40"),
            "incineration": Decimal("0.05"),
            "recycling": Decimal("0.50"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "batteries": {
        "ef_kgco2e_per_kg": Decimal("1.20"),
        "description": "Primary and rechargeable batteries (Li-ion, NiMH, lead-acid)",
        "treatment_mix": {
            "landfill": Decimal("0.15"),
            "incineration": Decimal("0.05"),
            "recycling": Decimal("0.75"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "tires": {
        "ef_kgco2e_per_kg": Decimal("0.95"),
        "description": "Passenger and commercial vehicle tires",
        "treatment_mix": {
            "landfill": Decimal("0.10"),
            "incineration": Decimal("0.35"),
            "recycling": Decimal("0.50"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "food_products": {
        "ef_kgco2e_per_kg": Decimal("1.50"),
        "description": "Food products at end of consumer life (unconsumed, expired)",
        "treatment_mix": {
            "landfill": Decimal("0.45"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.00"),
            "composting": Decimal("0.30"),
            "open_burning": Decimal("0.15"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "beverages": {
        "ef_kgco2e_per_kg": Decimal("0.30"),
        "description": "Beverage containers (glass, aluminium, PET, carton)",
        "treatment_mix": {
            "landfill": Decimal("0.25"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.60"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "paper_products": {
        "ef_kgco2e_per_kg": Decimal("0.48"),
        "description": "Paper, books, stationery, tissue, printed materials",
        "treatment_mix": {
            "landfill": Decimal("0.30"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.50"),
            "composting": Decimal("0.05"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "pharmaceuticals": {
        "ef_kgco2e_per_kg": Decimal("1.10"),
        "description": "Medicines, pharmaceutical products, medical devices",
        "treatment_mix": {
            "landfill": Decimal("0.15"),
            "incineration": Decimal("0.70"),
            "recycling": Decimal("0.10"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "toys_games": {
        "ef_kgco2e_per_kg": Decimal("0.60"),
        "description": "Toys, games, sporting goods (plastic, wood, fabric)",
        "treatment_mix": {
            "landfill": Decimal("0.45"),
            "incineration": Decimal("0.20"),
            "recycling": Decimal("0.25"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.10"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "glass_products": {
        "ef_kgco2e_per_kg": Decimal("0.25"),
        "description": "Glass containers, flat glass, specialty glass",
        "treatment_mix": {
            "landfill": Decimal("0.30"),
            "incineration": Decimal("0.00"),
            "recycling": Decimal("0.65"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "chemicals": {
        "ef_kgco2e_per_kg": Decimal("1.05"),
        "description": "Household chemicals, cleaners, paints, solvents",
        "treatment_mix": {
            "landfill": Decimal("0.20"),
            "incineration": Decimal("0.65"),
            "recycling": Decimal("0.10"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "personal_care": {
        "ef_kgco2e_per_kg": Decimal("0.50"),
        "description": "Cosmetics, hygiene products, personal care items",
        "treatment_mix": {
            "landfill": Decimal("0.40"),
            "incineration": Decimal("0.25"),
            "recycling": Decimal("0.30"),
            "composting": Decimal("0.00"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "garden_products": {
        "ef_kgco2e_per_kg": Decimal("0.42"),
        "description": "Garden supplies, pots, tools, soil products",
        "treatment_mix": {
            "landfill": Decimal("0.35"),
            "incineration": Decimal("0.10"),
            "recycling": Decimal("0.35"),
            "composting": Decimal("0.15"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "pet_products": {
        "ef_kgco2e_per_kg": Decimal("0.58"),
        "description": "Pet food containers, accessories, toys, bedding",
        "treatment_mix": {
            "landfill": Decimal("0.50"),
            "incineration": Decimal("0.15"),
            "recycling": Decimal("0.20"),
            "composting": Decimal("0.10"),
            "open_burning": Decimal("0.05"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
    "mixed_products": {
        "ef_kgco2e_per_kg": Decimal("0.587"),
        "description": "Mixed or unclassified products (global default fallback)",
        "treatment_mix": {
            "landfill": Decimal("0.35"),
            "incineration": Decimal("0.15"),
            "recycling": Decimal("0.35"),
            "composting": Decimal("0.05"),
            "open_burning": Decimal("0.10"),
        },
        "source": "epa_warm_v16_defra_blend",
        "source_year": 2025,
    },
}

# ==============================================================================
# DATA TABLE: REGIONAL ADJUSTMENT FACTORS
# ==============================================================================
# Adjusts composite EFs based on regional waste infrastructure differences.
# US has higher landfill rate (53%), EU has higher recycling (46%),
# Japan has high incineration with WtE (77%), etc.
# Source: OECD Municipal Waste Statistics 2024, World Bank What a Waste 2.0.

REGIONAL_ADJUSTMENT_FACTORS: Dict[str, Dict[str, Any]] = {
    "US": {
        "factor": Decimal("1.05"),
        "description": "United States - higher landfill share (53%)",
        "landfill_share": Decimal("0.53"),
        "recycling_share": Decimal("0.32"),
        "incineration_share": Decimal("0.12"),
        "composting_share": Decimal("0.03"),
    },
    "EU": {
        "factor": Decimal("0.85"),
        "description": "European Union - higher recycling and WtE share",
        "landfill_share": Decimal("0.24"),
        "recycling_share": Decimal("0.46"),
        "incineration_share": Decimal("0.26"),
        "composting_share": Decimal("0.04"),
    },
    "JP": {
        "factor": Decimal("0.90"),
        "description": "Japan - very high incineration with energy recovery (77%)",
        "landfill_share": Decimal("0.01"),
        "recycling_share": Decimal("0.20"),
        "incineration_share": Decimal("0.77"),
        "composting_share": Decimal("0.02"),
    },
    "CN": {
        "factor": Decimal("1.15"),
        "description": "China - high landfill share, growing incineration",
        "landfill_share": Decimal("0.52"),
        "recycling_share": Decimal("0.20"),
        "incineration_share": Decimal("0.25"),
        "composting_share": Decimal("0.03"),
    },
    "IN": {
        "factor": Decimal("1.30"),
        "description": "India - high open dumping, low recycling infrastructure",
        "landfill_share": Decimal("0.70"),
        "recycling_share": Decimal("0.10"),
        "incineration_share": Decimal("0.05"),
        "composting_share": Decimal("0.15"),
    },
    "BR": {
        "factor": Decimal("1.20"),
        "description": "Brazil - significant open dumping, growing recycling",
        "landfill_share": Decimal("0.60"),
        "recycling_share": Decimal("0.13"),
        "incineration_share": Decimal("0.02"),
        "composting_share": Decimal("0.25"),
    },
    "AU": {
        "factor": Decimal("1.00"),
        "description": "Australia - moderate landfill and recycling",
        "landfill_share": Decimal("0.40"),
        "recycling_share": Decimal("0.42"),
        "incineration_share": Decimal("0.03"),
        "composting_share": Decimal("0.15"),
    },
    "KR": {
        "factor": Decimal("0.88"),
        "description": "South Korea - high recycling (59%), extended producer responsibility",
        "landfill_share": Decimal("0.10"),
        "recycling_share": Decimal("0.59"),
        "incineration_share": Decimal("0.25"),
        "composting_share": Decimal("0.06"),
    },
    "DE": {
        "factor": Decimal("0.82"),
        "description": "Germany - near-zero landfill, very high recycling",
        "landfill_share": Decimal("0.01"),
        "recycling_share": Decimal("0.67"),
        "incineration_share": Decimal("0.30"),
        "composting_share": Decimal("0.02"),
    },
    "GB": {
        "factor": Decimal("0.92"),
        "description": "United Kingdom - growing recycling, declining landfill",
        "landfill_share": Decimal("0.24"),
        "recycling_share": Decimal("0.44"),
        "incineration_share": Decimal("0.28"),
        "composting_share": Decimal("0.04"),
    },
    "CA": {
        "factor": Decimal("1.02"),
        "description": "Canada - moderate landfill and recycling rates",
        "landfill_share": Decimal("0.45"),
        "recycling_share": Decimal("0.35"),
        "incineration_share": Decimal("0.10"),
        "composting_share": Decimal("0.10"),
    },
    "GLOBAL": {
        "factor": Decimal("1.00"),
        "description": "Global average (no adjustment)",
        "landfill_share": Decimal("0.37"),
        "recycling_share": Decimal("0.19"),
        "incineration_share": Decimal("0.11"),
        "composting_share": Decimal("0.06"),
    },
}

# ==============================================================================
# DATA TABLE: DEFAULT PRODUCT WEIGHTS
# ==============================================================================
# Estimated average weight per unit (kg) when actual weight is not provided.
# Source: Industry average data, EPA product stewardship guidance.

DEFAULT_PRODUCT_WEIGHTS: Dict[str, Decimal] = {
    "electronics": Decimal("0.50"),
    "appliances": Decimal("25.00"),
    "furniture": Decimal("30.00"),
    "packaging": Decimal("0.10"),
    "clothing": Decimal("0.40"),
    "automotive_parts": Decimal("5.00"),
    "building_materials": Decimal("20.00"),
    "batteries": Decimal("0.05"),
    "tires": Decimal("10.00"),
    "food_products": Decimal("0.50"),
    "beverages": Decimal("0.35"),
    "paper_products": Decimal("0.20"),
    "pharmaceuticals": Decimal("0.05"),
    "toys_games": Decimal("0.50"),
    "glass_products": Decimal("0.80"),
    "chemicals": Decimal("1.00"),
    "personal_care": Decimal("0.25"),
    "garden_products": Decimal("2.00"),
    "pet_products": Decimal("1.50"),
    "mixed_products": Decimal("1.00"),
}

# ==============================================================================
# GWP VALUES (IPCC Assessment Reports)
# ==============================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "ar4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "ar5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "ar6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
}

# ==============================================================================
# UNCERTAINTY PARAMETERS
# ==============================================================================

UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    "average_data_default": {
        "lower_pct": Decimal("0.30"),
        "upper_pct": Decimal("0.50"),
        "confidence_level": Decimal("0.95"),
    },
    "average_data_with_regional": {
        "lower_pct": Decimal("0.25"),
        "upper_pct": Decimal("0.45"),
        "confidence_level": Decimal("0.95"),
    },
}

# DQI score assignments for average-data method
DQI_AVERAGE_DATA: Dict[str, int] = {
    "temporal": 3,
    "geographical": 3,
    "technological": 4,
    "completeness": 3,
    "reliability": 4,
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _round_decimal(value: Decimal, precision: int = PRECISION) -> Decimal:
    """
    Round a Decimal to the specified number of decimal places.

    Args:
        value: Decimal value to round.
        precision: Number of decimal places.

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * precision
    return value.quantize(Decimal(quantize_str), rounding=ROUNDING)


def _compute_hash(data: str) -> str:
    """
    Compute SHA-256 hash for provenance tracking.

    Args:
        data: String data to hash.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "eol_avg") -> str:
    """
    Generate a unique identifier with prefix.

    Args:
        prefix: Identifier prefix.

    Returns:
        Unique ID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _safe_decimal(value: Any) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, or Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If conversion fails.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert {value!r} (type={type(value).__name__}) to Decimal"
        ) from exc


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class AverageDataCalculatorEngine:
    """
    Engine 3: Average-data EOL emissions calculator (Method B).

    Implements the average-data calculation method for GHG Protocol Scope 3
    Category 12 (End-of-Life Treatment of Sold Products). This method uses
    pre-mixed composite emission factors by product category that blend
    typical treatment scenarios into a single kgCO2e/kg factor.

    Formula:
        E_total = SUM_products [ units_sold x weight_per_unit x composite_EF ]

    The composite EF is resolved through a fallback hierarchy:
        1. Product-specific EF (custom override)
        2. Product category EF (from COMPOSITE_EOL_EF table)
        3. mixed_products fallback (0.587 kgCO2e/kg)

    Regional adjustments modify the composite EF to account for differences
    in waste infrastructure (e.g., EU has 46% recycling vs US 32%).

    Thread Safety:
        Singleton pattern with threading.RLock() and double-checked locking.
        All mutable state is protected by the lock.

    Zero-Hallucination:
        All calculations are deterministic Python Decimal arithmetic.
        No LLM calls for any numeric computation.

    Attributes:
        _config: Configuration from get_config().
        _metrics: Prometheus metrics from get_metrics().
        _calculation_count: Running count of calculations performed.
        _batch_count: Running count of batch calculations.

    Example:
        >>> engine = AverageDataCalculatorEngine.get_instance()
        >>> result = engine.calculate(
        ...     products=[{"category": "electronics", "units_sold": 500,
        ...                "weight_per_unit_kg": Decimal("0.3")}],
        ...     org_id="ORG-001", year=2025)
        >>> result["total_co2e_kg"]
        Decimal('127.500000')
    """

    _instance: Optional["AverageDataCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "AverageDataCalculatorEngine":
        """Thread-safe singleton instantiation with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self, gwp_version: str = "ar5") -> None:
        """
        Initialize the AverageDataCalculatorEngine.

        Args:
            gwp_version: IPCC assessment report version for GWP values.
                One of 'ar4', 'ar5', 'ar6'. Default 'ar5'.
        """
        if self._initialized:
            return

        self._gwp_version: str = gwp_version
        gwp_table = GWP_VALUES.get(gwp_version, GWP_VALUES["ar5"])
        self._gwp_ch4: Decimal = gwp_table["ch4"]
        self._gwp_n2o: Decimal = gwp_table["n2o"]

        self._config = get_config()
        self._metrics = get_metrics()
        self._calculation_count: int = 0
        self._batch_count: int = 0
        self._count_lock: threading.RLock = threading.RLock()

        self._initialized: bool = True

        logger.info(
            "AverageDataCalculatorEngine initialized: engine=%s, version=%s, "
            "gwp=%s, categories=%d, regions=%d",
            ENGINE_ID,
            ENGINE_VERSION,
            gwp_version,
            len(COMPOSITE_EOL_EF),
            len(REGIONAL_ADJUSTMENT_FACTORS),
        )

    # ==========================================================================
    # SINGLETON MANAGEMENT
    # ==========================================================================

    @classmethod
    def get_instance(cls, gwp_version: str = "ar5") -> "AverageDataCalculatorEngine":
        """
        Get singleton instance with thread-safe double-checked locking.

        Args:
            gwp_version: IPCC GWP version (only used on first instantiation).

        Returns:
            AverageDataCalculatorEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls(gwp_version=gwp_version)
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton instance. Used in testing only.

        Thread Safety:
            Protected by the class-level RLock.
        """
        with cls._lock:
            cls._instance = None
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _increment_calculation_count(self) -> int:
        """
        Increment and return the calculation counter thread-safely.

        Returns:
            Updated calculation count.
        """
        with self._count_lock:
            self._calculation_count += 1
            return self._calculation_count

    def _increment_batch_count(self) -> int:
        """
        Increment and return the batch counter thread-safely.

        Returns:
            Updated batch count.
        """
        with self._count_lock:
            self._batch_count += 1
            return self._batch_count

    def _record_metrics(
        self,
        category: str,
        co2e_kg: Decimal,
        duration: float,
        status: str,
        method: str = "average_data",
    ) -> None:
        """
        Record Prometheus metrics for calculation.

        Args:
            category: Product category.
            co2e_kg: Calculated emissions in kgCO2e.
            duration: Calculation duration in seconds.
            status: Calculation status ('success' or 'error').
            method: Calculation method identifier.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_calculation(
                engine=ENGINE_ID,
                method=method,
                category=category,
                co2e_kg=float(co2e_kg),
                duration=duration,
                status=status,
            )
        except Exception as exc:
            logger.debug("Metrics recording failed (non-critical): %s", exc)

    # ==========================================================================
    # COMPOSITE EMISSION FACTOR RESOLUTION
    # ==========================================================================

    def get_composite_ef(self, product_category: str) -> Decimal:
        """
        Get composite end-of-life emission factor for a product category.

        Resolution hierarchy:
            1. Exact match in COMPOSITE_EOL_EF table
            2. Case-insensitive match
            3. Fallback to 'mixed_products' (0.587 kgCO2e/kg)

        Args:
            product_category: Product category identifier (e.g., 'electronics').

        Returns:
            Composite EF in kgCO2e/kg.

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
            >>> engine.get_composite_ef("electronics")
            Decimal('0.85')
            >>> engine.get_composite_ef("unknown_product")
            Decimal('0.587')
        """
        # Step 1: Exact match
        entry = COMPOSITE_EOL_EF.get(product_category)
        if entry is not None:
            ef = entry["ef_kgco2e_per_kg"]
            logger.debug(
                "Composite EF resolved: category=%s, ef=%s kgCO2e/kg (exact match)",
                product_category, ef,
            )
            return ef

        # Step 2: Case-insensitive match
        category_lower = product_category.lower().strip()
        for key, val in COMPOSITE_EOL_EF.items():
            if key.lower() == category_lower:
                ef = val["ef_kgco2e_per_kg"]
                logger.debug(
                    "Composite EF resolved: category=%s, ef=%s kgCO2e/kg "
                    "(case-insensitive match to '%s')",
                    product_category, ef, key,
                )
                return ef

        # Step 3: Fallback to mixed_products
        fallback_ef = COMPOSITE_EOL_EF["mixed_products"]["ef_kgco2e_per_kg"]
        logger.warning(
            "Composite EF not found for category '%s'; using mixed_products "
            "fallback EF=%s kgCO2e/kg",
            product_category, fallback_ef,
        )
        return fallback_ef

    def get_treatment_mix(self, product_category: str) -> Dict[str, Decimal]:
        """
        Get the treatment mix breakdown for a product category.

        Args:
            product_category: Product category identifier.

        Returns:
            Dictionary mapping treatment type to fraction (0-1).

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
            >>> mix = engine.get_treatment_mix("electronics")
            >>> mix["recycling"]
            Decimal('0.55')
        """
        entry = COMPOSITE_EOL_EF.get(product_category)
        if entry is None:
            category_lower = product_category.lower().strip()
            for key, val in COMPOSITE_EOL_EF.items():
                if key.lower() == category_lower:
                    entry = val
                    break

        if entry is None:
            entry = COMPOSITE_EOL_EF["mixed_products"]

        return dict(entry.get("treatment_mix", {}))

    # ==========================================================================
    # REGIONAL ADJUSTMENT
    # ==========================================================================

    def get_regional_adjustment(self, region: str) -> Decimal:
        """
        Get the regional adjustment factor for a given region.

        Resolution hierarchy:
            1. Exact match in REGIONAL_ADJUSTMENT_FACTORS
            2. Case-insensitive match
            3. Fallback to GLOBAL (1.00)

        Args:
            region: Region code (e.g., 'US', 'EU', 'JP', 'GLOBAL').

        Returns:
            Regional adjustment factor (Decimal).

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
            >>> engine.get_regional_adjustment("EU")
            Decimal('0.85')
            >>> engine.get_regional_adjustment("XX")
            Decimal('1.00')
        """
        entry = REGIONAL_ADJUSTMENT_FACTORS.get(region)
        if entry is not None:
            factor = entry["factor"]
            logger.debug(
                "Regional adjustment: region=%s, factor=%s (exact match)",
                region, factor,
            )
            return factor

        region_upper = region.upper().strip()
        for key, val in REGIONAL_ADJUSTMENT_FACTORS.items():
            if key.upper() == region_upper:
                factor = val["factor"]
                logger.debug(
                    "Regional adjustment: region=%s, factor=%s "
                    "(case-insensitive match to '%s')",
                    region, factor, key,
                )
                return factor

        fallback = REGIONAL_ADJUSTMENT_FACTORS["GLOBAL"]["factor"]
        logger.warning(
            "Regional adjustment not found for '%s'; using GLOBAL fallback=%s",
            region, fallback,
        )
        return fallback

    # ==========================================================================
    # WEIGHT ESTIMATION
    # ==========================================================================

    def estimate_weight(
        self,
        product_category: str,
        units: int,
    ) -> Decimal:
        """
        Estimate total weight from units sold using default product weights.

        Used when the reporting company does not know the weight per unit.
        The default weights are conservative industry averages.

        Args:
            product_category: Product category identifier.
            units: Number of units sold.

        Returns:
            Estimated total weight in kg.

        Raises:
            ValueError: If units <= 0.

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
            >>> engine.estimate_weight("electronics", 1000)
            Decimal('500.00')
        """
        if units <= 0:
            raise ValueError(f"units must be > 0, got {units}")

        weight_per_unit = DEFAULT_PRODUCT_WEIGHTS.get(product_category)
        if weight_per_unit is None:
            category_lower = product_category.lower().strip()
            for key, val in DEFAULT_PRODUCT_WEIGHTS.items():
                if key.lower() == category_lower:
                    weight_per_unit = val
                    break

        if weight_per_unit is None:
            weight_per_unit = DEFAULT_PRODUCT_WEIGHTS["mixed_products"]
            logger.warning(
                "Default weight not found for '%s'; using mixed_products=%s kg",
                product_category, weight_per_unit,
            )

        total_weight = weight_per_unit * Decimal(str(units))
        logger.debug(
            "Weight estimate: category=%s, units=%d, weight_per_unit=%s kg, "
            "total=%s kg",
            product_category, units, weight_per_unit, total_weight,
        )
        return _round_decimal(total_weight, 2)

    # ==========================================================================
    # CORE CALCULATION
    # ==========================================================================

    def calculate(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate average-data EOL emissions for a list of sold products.

        Formula per product:
            emissions_kg = units_sold x weight_per_unit_kg x composite_EF

        If weight_per_unit_kg is not provided, it is estimated from the
        product category using DEFAULT_PRODUCT_WEIGHTS.

        Args:
            products: List of product dictionaries, each containing:
                - product_id (str, optional): Product identifier.
                - category (str): Product category (e.g., 'electronics').
                - units_sold (int): Number of units sold in reporting year.
                - weight_per_unit_kg (Decimal, optional): Weight per unit in kg.
                - custom_ef (Decimal, optional): Custom override EF (kgCO2e/kg).
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with:
                - calculation_id (str)
                - org_id (str)
                - year (int)
                - method (str): 'average_data'
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - total_weight_kg (Decimal)
                - product_results (list): Per-product breakdown
                - dqi_score (dict): Data quality indicator scores
                - uncertainty (dict): Uncertainty range
                - provenance_hash (str): SHA-256 audit hash
                - processing_time_ms (float)

        Raises:
            ValueError: If products list is empty or invalid.
        """
        start_time = time.monotonic()
        calc_id = _generate_id("eol_avg")

        logger.info(
            "AverageDataCalculatorEngine.calculate: calc_id=%s, org=%s, "
            "year=%d, products=%d",
            calc_id, org_id, year, len(products),
        )

        # Validate inputs
        self._validate_product_list(products, org_id, year)

        # Calculate per product
        product_results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_weight_kg = _ZERO
        errors: List[Dict[str, str]] = []

        for idx, product in enumerate(products):
            try:
                result = self._calculate_single_product(product, idx)
                product_results.append(result)
                total_co2e_kg += result["co2e_kg"]
                total_weight_kg += result["weight_kg"]
            except Exception as exc:
                product_id = product.get("product_id", f"product_{idx}")
                error_msg = str(exc)
                logger.error(
                    "Product calculation failed: product=%s, error=%s",
                    product_id, error_msg,
                )
                errors.append({
                    "product_id": product_id,
                    "error": error_msg,
                })

        # Round totals
        total_co2e_kg = _round_decimal(total_co2e_kg)
        total_co2e_tonnes = _round_decimal(total_co2e_kg * TONNES_PER_KG)
        total_weight_kg = _round_decimal(total_weight_kg, 2)

        # Compute DQI and uncertainty
        dqi_score = self.compute_dqi_score()
        uncertainty = self.compute_uncertainty(total_co2e_kg)

        # Compute provenance hash
        provenance_data = (
            f"{calc_id}|{org_id}|{year}|average_data|"
            f"{total_co2e_kg}|{total_weight_kg}|"
            f"{len(product_results)}|{len(errors)}"
        )
        provenance_hash = _compute_hash(provenance_data)

        # Record metrics
        duration = time.monotonic() - start_time
        self._record_metrics(
            category="batch",
            co2e_kg=total_co2e_kg,
            duration=duration,
            status="success" if not errors else "partial",
        )

        count = self._increment_calculation_count()

        result = {
            "calculation_id": calc_id,
            "org_id": org_id,
            "year": year,
            "method": "average_data",
            "method_description": (
                "GHG Protocol Scope 3 Category 12 average-data method (Method B). "
                "Uses pre-mixed composite EFs by product category."
            ),
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_weight_kg": total_weight_kg,
            "product_count": len(products),
            "success_count": len(product_results),
            "error_count": len(errors),
            "product_results": product_results,
            "errors": errors,
            "dqi_score": dqi_score,
            "uncertainty": uncertainty,
            "gwp_version": self._gwp_version,
            "provenance_hash": provenance_hash,
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "calculation_number": count,
            "processing_time_ms": round(duration * 1000, 2),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "AverageDataCalculatorEngine.calculate complete: calc_id=%s, "
            "total_co2e_kg=%s, total_co2e_tonnes=%s, products=%d, "
            "errors=%d, duration_ms=%.2f",
            calc_id, total_co2e_kg, total_co2e_tonnes,
            len(product_results), len(errors), duration * 1000,
        )

        return result

    def calculate_with_regional_adjustment(
        self,
        products: List[Dict[str, Any]],
        region: str,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Calculate average-data EOL emissions with regional adjustment.

        Applies a regional multiplier to account for differences in waste
        infrastructure between regions (e.g., EU has higher recycling).

        Formula:
            E_adjusted = E_base x regional_factor

        Args:
            products: List of product dictionaries (same format as calculate()).
            region: Region code (e.g., 'US', 'EU', 'JP', 'CN', 'IN', 'GLOBAL').
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Dictionary with same structure as calculate() plus regional fields.
        """
        start_time = time.monotonic()

        logger.info(
            "AverageDataCalculatorEngine.calculate_with_regional_adjustment: "
            "region=%s, org=%s, year=%d, products=%d",
            region, org_id, year, len(products),
        )

        # Get base calculation
        base_result = self.calculate(products, org_id, year)

        # Apply regional adjustment
        regional_factor = self.get_regional_adjustment(region)
        adjusted_co2e_kg = _round_decimal(
            base_result["total_co2e_kg"] * regional_factor
        )
        adjusted_co2e_tonnes = _round_decimal(adjusted_co2e_kg * TONNES_PER_KG)

        # Adjust each product result
        adjusted_products: List[Dict[str, Any]] = []
        for pr in base_result["product_results"]:
            adj_pr = dict(pr)
            adj_pr["co2e_kg_pre_adjustment"] = pr["co2e_kg"]
            adj_pr["co2e_kg"] = _round_decimal(pr["co2e_kg"] * regional_factor)
            adj_pr["co2e_tonnes"] = _round_decimal(
                adj_pr["co2e_kg"] * TONNES_PER_KG
            )
            adj_pr["regional_factor"] = regional_factor
            adj_pr["region"] = region
            adjusted_products.append(adj_pr)

        # Update uncertainty for regional method
        uncertainty = self.compute_uncertainty(
            adjusted_co2e_kg,
            uncertainty_type="average_data_with_regional",
        )

        # Recompute provenance
        provenance_data = (
            f"{base_result['calculation_id']}|{org_id}|{year}|"
            f"average_data_regional|{region}|{regional_factor}|"
            f"{adjusted_co2e_kg}|{base_result['total_weight_kg']}"
        )
        provenance_hash = _compute_hash(provenance_data)

        duration = time.monotonic() - start_time

        regional_info = REGIONAL_ADJUSTMENT_FACTORS.get(
            region,
            REGIONAL_ADJUSTMENT_FACTORS.get("GLOBAL", {}),
        )

        result = dict(base_result)
        result.update({
            "method": "average_data_regional",
            "method_description": (
                f"GHG Protocol Scope 3 Category 12 average-data method (Method B) "
                f"with regional adjustment for {region}."
            ),
            "total_co2e_kg_pre_adjustment": base_result["total_co2e_kg"],
            "total_co2e_kg": adjusted_co2e_kg,
            "total_co2e_tonnes": adjusted_co2e_tonnes,
            "region": region,
            "regional_factor": regional_factor,
            "regional_description": regional_info.get("description", ""),
            "regional_treatment_mix": {
                k: v for k, v in regional_info.items()
                if k.endswith("_share")
            },
            "product_results": adjusted_products,
            "uncertainty": uncertainty,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(duration * 1000, 2),
        })

        logger.info(
            "Regional adjustment complete: region=%s, factor=%s, "
            "base_co2e=%s, adjusted_co2e=%s",
            region, regional_factor,
            base_result["total_co2e_kg"], adjusted_co2e_kg,
        )

        return result

    # ==========================================================================
    # SINGLE PRODUCT CALCULATION
    # ==========================================================================

    def _calculate_single_product(
        self,
        product: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        """
        Calculate EOL emissions for a single product.

        Args:
            product: Product dictionary with category, units_sold, etc.
            index: Index in the products list (for error reporting).

        Returns:
            Per-product result dictionary.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        product_id = product.get("product_id", f"product_{index}")
        category = product.get("category", "mixed_products")
        units_sold = product.get("units_sold", 0)

        # Validate units_sold
        if not isinstance(units_sold, int) or units_sold <= 0:
            try:
                units_sold = int(units_sold)
                if units_sold <= 0:
                    raise ValueError("units_sold must be > 0")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Product '{product_id}': units_sold must be a positive "
                    f"integer, got {units_sold!r}"
                ) from exc

        # Resolve weight per unit
        weight_per_unit_kg = product.get("weight_per_unit_kg")
        weight_estimated = False
        if weight_per_unit_kg is None:
            weight_per_unit_kg = self._get_default_weight(category)
            weight_estimated = True
        else:
            weight_per_unit_kg = _safe_decimal(weight_per_unit_kg)
            if weight_per_unit_kg <= _ZERO:
                raise ValueError(
                    f"Product '{product_id}': weight_per_unit_kg must be > 0, "
                    f"got {weight_per_unit_kg}"
                )

        # Resolve emission factor
        custom_ef = product.get("custom_ef")
        if custom_ef is not None:
            ef = _safe_decimal(custom_ef)
            ef_source = "custom_override"
        else:
            ef = self.get_composite_ef(category)
            ef_source = "composite_eol_ef"

        # Calculate total weight
        total_weight_kg = _round_decimal(
            weight_per_unit_kg * Decimal(str(units_sold)), 2
        )

        # Calculate emissions: weight_kg x EF
        co2e_kg = _round_decimal(total_weight_kg * ef)
        co2e_tonnes = _round_decimal(co2e_kg * TONNES_PER_KG)

        # Get treatment mix for breakdown reporting
        treatment_mix = self.get_treatment_mix(category)

        # Provenance for this product
        prov_data = (
            f"{product_id}|{category}|{units_sold}|"
            f"{weight_per_unit_kg}|{ef}|{co2e_kg}"
        )
        product_hash = _compute_hash(prov_data)

        result = {
            "product_id": product_id,
            "category": category,
            "units_sold": units_sold,
            "weight_per_unit_kg": weight_per_unit_kg,
            "weight_estimated": weight_estimated,
            "weight_kg": total_weight_kg,
            "weight_tonnes": _round_decimal(total_weight_kg * TONNES_PER_KG),
            "ef_kgco2e_per_kg": ef,
            "ef_source": ef_source,
            "co2e_kg": co2e_kg,
            "co2e_tonnes": co2e_tonnes,
            "treatment_mix": treatment_mix,
            "provenance_hash": product_hash,
        }

        logger.debug(
            "Product calculated: id=%s, category=%s, units=%d, "
            "weight=%s kg, ef=%s, co2e=%s kg",
            product_id, category, units_sold,
            total_weight_kg, ef, co2e_kg,
        )

        return result

    def _get_default_weight(self, product_category: str) -> Decimal:
        """
        Get default weight for a product category.

        Args:
            product_category: Product category identifier.

        Returns:
            Default weight in kg.
        """
        weight = DEFAULT_PRODUCT_WEIGHTS.get(product_category)
        if weight is not None:
            return weight

        category_lower = product_category.lower().strip()
        for key, val in DEFAULT_PRODUCT_WEIGHTS.items():
            if key.lower() == category_lower:
                return val

        return DEFAULT_PRODUCT_WEIGHTS["mixed_products"]

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def _validate_product_list(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> None:
        """
        Validate the input product list.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            year: Reporting year.

        Raises:
            ValueError: If validation fails.
        """
        if not products:
            raise ValueError("products list must not be empty")

        if not isinstance(products, list):
            raise ValueError(
                f"products must be a list, got {type(products).__name__}"
            )

        if not org_id or not isinstance(org_id, str):
            raise ValueError(
                f"org_id must be a non-empty string, got {org_id!r}"
            )

        if not isinstance(year, int) or year < 2000 or year > 2100:
            raise ValueError(
                f"year must be an integer between 2000 and 2100, got {year}"
            )

        for idx, product in enumerate(products):
            if not isinstance(product, dict):
                raise ValueError(
                    f"Product at index {idx} must be a dictionary, "
                    f"got {type(product).__name__}"
                )
            if "units_sold" not in product:
                product_id = product.get("product_id", f"product_{idx}")
                raise ValueError(
                    f"Product '{product_id}' missing required field 'units_sold'"
                )

    def validate_inputs(
        self,
        products: List[Dict[str, Any]],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Validate inputs and return detailed validation result.

        Args:
            products: List of product dictionaries.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Validation result dictionary with is_valid, errors, warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        valid_count = 0
        invalid_count = 0

        # Validate org_id
        if not org_id or not isinstance(org_id, str):
            errors.append("org_id must be a non-empty string")

        # Validate year
        if not isinstance(year, int) or year < 2000 or year > 2100:
            errors.append("year must be an integer between 2000 and 2100")

        # Validate products list
        if not products:
            errors.append("products list must not be empty")
        elif not isinstance(products, list):
            errors.append("products must be a list")
        else:
            for idx, product in enumerate(products):
                product_id = product.get("product_id", f"product_{idx}")
                if not isinstance(product, dict):
                    errors.append(
                        f"Product at index {idx} must be a dictionary"
                    )
                    invalid_count += 1
                    continue

                # Check required field
                if "units_sold" not in product:
                    errors.append(
                        f"Product '{product_id}' missing 'units_sold'"
                    )
                    invalid_count += 1
                    continue

                # Check units_sold value
                try:
                    units = int(product["units_sold"])
                    if units <= 0:
                        errors.append(
                            f"Product '{product_id}': units_sold must be > 0"
                        )
                        invalid_count += 1
                        continue
                except (TypeError, ValueError):
                    errors.append(
                        f"Product '{product_id}': units_sold must be numeric"
                    )
                    invalid_count += 1
                    continue

                # Check category
                category = product.get("category", "mixed_products")
                if category not in COMPOSITE_EOL_EF:
                    category_lower = category.lower().strip()
                    found = any(
                        k.lower() == category_lower
                        for k in COMPOSITE_EOL_EF
                    )
                    if not found:
                        warnings.append(
                            f"Product '{product_id}': category '{category}' "
                            f"not in standard categories; will use fallback EF"
                        )

                # Check weight
                weight = product.get("weight_per_unit_kg")
                if weight is not None:
                    try:
                        w = Decimal(str(weight))
                        if w <= _ZERO:
                            errors.append(
                                f"Product '{product_id}': "
                                f"weight_per_unit_kg must be > 0"
                            )
                            invalid_count += 1
                            continue
                    except (InvalidOperation, ValueError):
                        errors.append(
                            f"Product '{product_id}': "
                            f"weight_per_unit_kg must be numeric"
                        )
                        invalid_count += 1
                        continue
                else:
                    warnings.append(
                        f"Product '{product_id}': weight_per_unit_kg not "
                        f"provided; will use default for category '{category}'"
                    )

                valid_count += 1

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "valid_product_count": valid_count,
            "invalid_product_count": invalid_count,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    # ==========================================================================
    # DATA QUALITY INDICATOR
    # ==========================================================================

    def compute_dqi_score(self) -> Dict[str, Any]:
        """
        Compute data quality indicator scores for average-data method.

        Average-data is Tier 2: scores are generally 3-4 on a 1-5 scale.
        Lower scores indicate higher quality.

        Returns:
            Dictionary with dimension scores, composite score, classification,
            and tier.

        Example:
            >>> engine = AverageDataCalculatorEngine.get_instance()
            >>> dqi = engine.compute_dqi_score()
            >>> dqi["tier"]
            'tier_2'
        """
        dimension_scores = dict(DQI_AVERAGE_DATA)
        composite = Decimal(str(
            sum(dimension_scores.values())
        )) / Decimal(str(len(dimension_scores)))
        composite = _round_decimal(composite, 2)

        # Classification
        if composite <= Decimal("1.5"):
            classification = "very_good"
        elif composite <= Decimal("2.5"):
            classification = "good"
        elif composite <= Decimal("3.5"):
            classification = "fair"
        elif composite <= Decimal("4.5"):
            classification = "poor"
        else:
            classification = "very_poor"

        return {
            "method": "average_data",
            "tier": "tier_2",
            "dimension_scores": dimension_scores,
            "composite_score": composite,
            "classification": classification,
            "description": (
                "Average-data method uses composite emission factors by product "
                "category. Data quality is Tier 2 (moderate). Consider upgrading "
                "to producer-specific (Tier 1) for material product lines."
            ),
            "improvement_recommendations": [
                "Obtain EPD data from suppliers for top-selling products",
                "Collect product weight data instead of using defaults",
                "Identify regional treatment scenarios for key markets",
                "Request producer-specific EOL scenario data for high-volume SKUs",
            ],
        }

    # ==========================================================================
    # UNCERTAINTY
    # ==========================================================================

    def compute_uncertainty(
        self,
        total_co2e_kg: Decimal,
        uncertainty_type: str = "average_data_default",
    ) -> Dict[str, Any]:
        """
        Compute uncertainty range for average-data calculation.

        Average-data uncertainty is +/-30-50% at 95% confidence.

        Args:
            total_co2e_kg: Total calculated emissions in kgCO2e.
            uncertainty_type: One of 'average_data_default',
                'average_data_with_regional'.

        Returns:
            Dictionary with lower_bound, upper_bound, range_pct, etc.
        """
        params = UNCERTAINTY_RANGES.get(
            uncertainty_type,
            UNCERTAINTY_RANGES["average_data_default"],
        )

        lower_pct = params["lower_pct"]
        upper_pct = params["upper_pct"]
        confidence = params["confidence_level"]

        lower_bound = _round_decimal(
            total_co2e_kg * (_ONE - lower_pct)
        )
        upper_bound = _round_decimal(
            total_co2e_kg * (_ONE + upper_pct)
        )

        # Symmetric mid-range for reporting
        mid_pct = _round_decimal((lower_pct + upper_pct) / Decimal("2"), 2)

        return {
            "method": "ipcc_default",
            "uncertainty_type": uncertainty_type,
            "total_co2e_kg": total_co2e_kg,
            "lower_bound_co2e_kg": lower_bound,
            "upper_bound_co2e_kg": upper_bound,
            "lower_pct": lower_pct,
            "upper_pct": upper_pct,
            "symmetric_pct": mid_pct,
            "confidence_level": confidence,
            "description": (
                f"Uncertainty range for {uncertainty_type}: "
                f"-{lower_pct * 100}% / +{upper_pct * 100}% "
                f"at {confidence * 100}% confidence level."
            ),
        }

    # ==========================================================================
    # BATCH CALCULATION
    # ==========================================================================

    def calculate_batch(
        self,
        product_batches: List[Dict[str, Any]],
        org_id: str,
        year: int,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for multiple product batches.

        Each batch is a group of products that can be independently calculated.
        This method supports optional regional adjustment.

        Args:
            product_batches: List of batch dictionaries, each containing:
                - batch_id (str): Batch identifier.
                - products (list): List of product dicts.
            org_id: Organization identifier.
            year: Reporting year.
            region: Optional region code for adjustment.

        Returns:
            Batch result dictionary with per-batch and aggregate totals.
        """
        start_time = time.monotonic()
        batch_id = _generate_id("eol_avg_batch")

        logger.info(
            "AverageDataCalculatorEngine.calculate_batch: batch_id=%s, "
            "batches=%d, region=%s",
            batch_id, len(product_batches), region,
        )

        batch_results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        total_weight_kg = _ZERO
        total_products = 0
        total_errors = 0

        for batch in product_batches:
            sub_batch_id = batch.get("batch_id", _generate_id("sub"))
            products = batch.get("products", [])
            if not products:
                logger.warning(
                    "Empty batch skipped: batch_id=%s", sub_batch_id
                )
                continue

            try:
                if region:
                    result = self.calculate_with_regional_adjustment(
                        products, region, org_id, year
                    )
                else:
                    result = self.calculate(products, org_id, year)

                result["batch_id"] = sub_batch_id
                batch_results.append(result)
                total_co2e_kg += result["total_co2e_kg"]
                total_weight_kg += result["total_weight_kg"]
                total_products += result["success_count"]
                total_errors += result["error_count"]

            except Exception as exc:
                logger.error(
                    "Batch calculation failed: batch_id=%s, error=%s",
                    sub_batch_id, str(exc),
                )
                batch_results.append({
                    "batch_id": sub_batch_id,
                    "status": "error",
                    "error": str(exc),
                })
                total_errors += 1

        total_co2e_kg = _round_decimal(total_co2e_kg)
        total_co2e_tonnes = _round_decimal(total_co2e_kg * TONNES_PER_KG)
        total_weight_kg = _round_decimal(total_weight_kg, 2)

        provenance_data = (
            f"{batch_id}|{org_id}|{year}|batch|"
            f"{total_co2e_kg}|{total_weight_kg}|"
            f"{len(batch_results)}|{total_errors}"
        )
        provenance_hash = _compute_hash(provenance_data)

        duration = time.monotonic() - start_time
        self._increment_batch_count()

        return {
            "batch_id": batch_id,
            "org_id": org_id,
            "year": year,
            "region": region,
            "method": "average_data_batch",
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_weight_kg": total_weight_kg,
            "batch_count": len(product_batches),
            "total_product_count": total_products,
            "total_error_count": total_errors,
            "batch_results": batch_results,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(duration * 1000, 2),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ==========================================================================
    # CATEGORY SUMMARY
    # ==========================================================================

    def summarize_by_category(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Summarize calculation results by product category.

        Args:
            results: Output from calculate() or calculate_with_regional_adjustment().

        Returns:
            Summary dictionary with per-category totals and percentages.
        """
        product_results = results.get("product_results", [])
        if not product_results:
            return {
                "categories": [],
                "total_co2e_kg": _ZERO,
                "total_weight_kg": _ZERO,
            }

        category_map: Dict[str, Dict[str, Decimal]] = {}
        for pr in product_results:
            cat = pr.get("category", "mixed_products")
            if cat not in category_map:
                category_map[cat] = {
                    "co2e_kg": _ZERO,
                    "weight_kg": _ZERO,
                    "units": _ZERO,
                    "product_count": _ZERO,
                }
            category_map[cat]["co2e_kg"] += pr.get("co2e_kg", _ZERO)
            category_map[cat]["weight_kg"] += pr.get("weight_kg", _ZERO)
            category_map[cat]["units"] += Decimal(str(pr.get("units_sold", 0)))
            category_map[cat]["product_count"] += _ONE

        total_co2e = sum(v["co2e_kg"] for v in category_map.values())

        categories = []
        for cat, vals in sorted(
            category_map.items(),
            key=lambda x: x[1]["co2e_kg"],
            reverse=True,
        ):
            pct = (
                _round_decimal(vals["co2e_kg"] / total_co2e * Decimal("100"), 2)
                if total_co2e > _ZERO
                else _ZERO
            )
            categories.append({
                "category": cat,
                "co2e_kg": _round_decimal(vals["co2e_kg"]),
                "weight_kg": _round_decimal(vals["weight_kg"], 2),
                "units": int(vals["units"]),
                "product_count": int(vals["product_count"]),
                "percentage_of_total": pct,
                "ef_kgco2e_per_kg": self.get_composite_ef(cat),
            })

        return {
            "categories": categories,
            "total_co2e_kg": _round_decimal(total_co2e),
            "total_weight_kg": _round_decimal(
                sum(v["weight_kg"] for v in category_map.values()), 2
            ),
            "category_count": len(categories),
        }

    # ==========================================================================
    # AVAILABLE CATEGORIES
    # ==========================================================================

    def list_categories(self) -> List[Dict[str, Any]]:
        """
        List all available product categories with their EFs.

        Returns:
            List of category dictionaries with name, EF, and description.
        """
        categories = []
        for key, entry in sorted(COMPOSITE_EOL_EF.items()):
            categories.append({
                "category": key,
                "ef_kgco2e_per_kg": entry["ef_kgco2e_per_kg"],
                "description": entry["description"],
                "source": entry["source"],
                "source_year": entry["source_year"],
                "default_weight_kg": DEFAULT_PRODUCT_WEIGHTS.get(key),
            })
        return categories

    def list_regions(self) -> List[Dict[str, Any]]:
        """
        List all available regions with their adjustment factors.

        Returns:
            List of region dictionaries with code, factor, and description.
        """
        regions = []
        for key, entry in sorted(REGIONAL_ADJUSTMENT_FACTORS.items()):
            regions.append({
                "region": key,
                "factor": entry["factor"],
                "description": entry["description"],
            })
        return regions

    # ==========================================================================
    # HEALTH CHECK
    # ==========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform engine health check.

        Validates that all data tables are loaded and consistent.

        Returns:
            Health check result dictionary.
        """
        checks: List[Dict[str, Any]] = []
        overall_healthy = True

        # Check 1: Composite EF table loaded
        ef_count = len(COMPOSITE_EOL_EF)
        ef_ok = ef_count >= 20
        checks.append({
            "check": "composite_ef_table",
            "status": "pass" if ef_ok else "fail",
            "detail": f"{ef_count} product categories loaded (expected >= 20)",
        })
        if not ef_ok:
            overall_healthy = False

        # Check 2: Regional adjustment table loaded
        region_count = len(REGIONAL_ADJUSTMENT_FACTORS)
        region_ok = region_count >= 12
        checks.append({
            "check": "regional_adjustment_table",
            "status": "pass" if region_ok else "fail",
            "detail": f"{region_count} regions loaded (expected >= 12)",
        })
        if not region_ok:
            overall_healthy = False

        # Check 3: Default weights table loaded
        weight_count = len(DEFAULT_PRODUCT_WEIGHTS)
        weight_ok = weight_count >= 20
        checks.append({
            "check": "default_weights_table",
            "status": "pass" if weight_ok else "fail",
            "detail": f"{weight_count} default weights loaded (expected >= 20)",
        })
        if not weight_ok:
            overall_healthy = False

        # Check 4: Mixed products fallback exists
        fallback_ok = "mixed_products" in COMPOSITE_EOL_EF
        checks.append({
            "check": "mixed_products_fallback",
            "status": "pass" if fallback_ok else "fail",
            "detail": "mixed_products fallback EF exists" if fallback_ok
                      else "MISSING mixed_products fallback",
        })
        if not fallback_ok:
            overall_healthy = False

        # Check 5: GLOBAL region fallback exists
        global_ok = "GLOBAL" in REGIONAL_ADJUSTMENT_FACTORS
        checks.append({
            "check": "global_region_fallback",
            "status": "pass" if global_ok else "fail",
            "detail": "GLOBAL region fallback exists" if global_ok
                      else "MISSING GLOBAL region fallback",
        })
        if not global_ok:
            overall_healthy = False

        # Check 6: EF values are positive
        negative_efs = [
            k for k, v in COMPOSITE_EOL_EF.items()
            if v["ef_kgco2e_per_kg"] <= _ZERO
        ]
        ef_positive_ok = len(negative_efs) == 0
        checks.append({
            "check": "ef_values_positive",
            "status": "pass" if ef_positive_ok else "fail",
            "detail": (
                "All EF values are positive"
                if ef_positive_ok
                else f"Negative EFs found: {negative_efs}"
            ),
        })
        if not ef_positive_ok:
            overall_healthy = False

        # Check 7: Regional factors are positive
        negative_regions = [
            k for k, v in REGIONAL_ADJUSTMENT_FACTORS.items()
            if v["factor"] <= _ZERO
        ]
        region_positive_ok = len(negative_regions) == 0
        checks.append({
            "check": "regional_factors_positive",
            "status": "pass" if region_positive_ok else "fail",
            "detail": (
                "All regional factors are positive"
                if region_positive_ok
                else f"Non-positive regional factors: {negative_regions}"
            ),
        })
        if not region_positive_ok:
            overall_healthy = False

        # Check 8: Treatment mix sums to 1.0
        invalid_mixes: List[str] = []
        for key, entry in COMPOSITE_EOL_EF.items():
            mix = entry.get("treatment_mix", {})
            if mix:
                total = sum(mix.values())
                if abs(total - _ONE) > Decimal("0.01"):
                    invalid_mixes.append(f"{key} (sum={total})")

        mix_ok = len(invalid_mixes) == 0
        checks.append({
            "check": "treatment_mix_sums",
            "status": "pass" if mix_ok else "warn",
            "detail": (
                "All treatment mixes sum to 1.0"
                if mix_ok
                else f"Invalid mixes: {invalid_mixes}"
            ),
        })

        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
            "calculation_count": self._calculation_count,
            "batch_count": self._batch_count,
            "categories_available": ef_count,
            "regions_available": region_count,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# ==============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ==============================================================================


def get_average_data_calculator(
    gwp_version: str = "ar5",
) -> AverageDataCalculatorEngine:
    """
    Get the singleton AverageDataCalculatorEngine instance.

    Args:
        gwp_version: IPCC GWP version ('ar4', 'ar5', 'ar6').

    Returns:
        AverageDataCalculatorEngine singleton instance.

    Example:
        >>> engine = get_average_data_calculator()
        >>> result = engine.calculate(
        ...     products=[{"category": "electronics", "units_sold": 100}],
        ...     org_id="ORG-001", year=2025)
    """
    return AverageDataCalculatorEngine.get_instance(gwp_version=gwp_version)


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Engine class
    "AverageDataCalculatorEngine",
    # Convenience function
    "get_average_data_calculator",
    # Data tables
    "COMPOSITE_EOL_EF",
    "REGIONAL_ADJUSTMENT_FACTORS",
    "DEFAULT_PRODUCT_WEIGHTS",
    "GWP_VALUES",
    "UNCERTAINTY_RANGES",
    "DQI_AVERAGE_DATA",
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "ENGINE_ID",
    "ENGINE_VERSION",
    "TABLE_PREFIX",
    "PRECISION",
]
