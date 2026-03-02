# -*- coding: utf-8 -*-
"""
EndOfLifeTreatmentPipelineEngine - AGENT-MRV-025 Engine 7

This module implements the EndOfLifeTreatmentPipelineEngine for End-of-Life Treatment
of Sold Products (GHG Protocol Scope 3 Category 12). It orchestrates a 10-stage pipeline
for complete end-of-life emissions calculation from raw input to compliant output with
full audit trail.

The 10 stages are:
1. VALIDATE: Input validation (category, units, weight, material composition, treatment)
2. CLASSIFY: Product classification, material BOM resolution, treatment pathway ID
3. NORMALIZE: Unit conversions (lbs->kg, tonnes->kg, oz->kg), fraction normalization
4. RESOLVE_EFS: Material x treatment EF lookup, regional adjustment, FOD/incineration params
5. CALCULATE: Method dispatch (waste-type-specific / average-data / producer-specific)
6. ALLOCATE: Proportional allocation across products, materials, treatment methods
7. AGGREGATE: Multi-dimensional aggregation (by treatment, material, category, region, period)
8. COMPLIANCE: Delegate to ComplianceCheckerEngine
9. PROVENANCE: Build provenance chain entries per stage
10. SEAL: SHA-256 hash of complete provenance chain

Built-in Data Tables:
    - 15 material x treatment emission factors
    - 20 default product BOMs (bill of materials)
    - 12 regional treatment mixes
    - Composite EFs for average-data method
    - Weight defaults per product category

Example:
    >>> from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
    ...     EndOfLifeTreatmentPipelineEngine
    ... )
    >>> engine = EndOfLifeTreatmentPipelineEngine()
    >>> result = engine.run_pipeline(
    ...     inputs={"products": [{"name": "Widget", "weight_kg": 0.5, "units_sold": 100000}]},
    ...     org_id="org-001",
    ...     year=2025,
    ... )
    >>> print(f"Total: {result['total_co2e_kg']} kgCO2e")

Module: greenlang.end_of_life_treatment.end_of_life_treatment_pipeline
Agent: AGENT-MRV-025
Version: 1.0.0
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "eol_treatment_pipeline_engine"
ENGINE_VERSION: str = "1.0.0"

_QUANT_2DP: Decimal = Decimal("0.01")
_QUANT_4DP: Decimal = Decimal("0.0001")
_QUANT_8DP: Decimal = Decimal("0.00000001")
ROUNDING: str = ROUND_HALF_UP

MAX_BATCH_SIZE: int = 5000

# Unit conversion factors
LBS_TO_KG: Decimal = Decimal("0.45359237")
TONNES_TO_KG: Decimal = Decimal("1000")
OZ_TO_KG: Decimal = Decimal("0.02834952")
SHORT_TONS_TO_KG: Decimal = Decimal("907.18474")


# ==============================================================================
# ENUMS
# ==============================================================================


class PipelineStage(str, Enum):
    """Pipeline stage identifiers."""

    VALIDATE = "validate"
    CLASSIFY = "classify"
    NORMALIZE = "normalize"
    RESOLVE_EFS = "resolve_efs"
    CALCULATE = "calculate"
    ALLOCATE = "allocate"
    AGGREGATE = "aggregate"
    COMPLIANCE = "compliance"
    PROVENANCE = "provenance"
    SEAL = "seal"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class CalculationMethod(str, Enum):
    """Calculation methods for Category 12."""

    WASTE_TYPE_SPECIFIC = "waste_type_specific"
    AVERAGE_DATA = "average_data"
    PRODUCER_SPECIFIC = "producer_specific"
    HYBRID = "hybrid"


class TreatmentPathway(str, Enum):
    """End-of-life treatment pathways."""

    LANDFILL = "landfill"
    INCINERATION = "incineration"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    OPEN_BURNING = "open_burning"
    WASTEWATER = "wastewater"
    WASTE_TO_ENERGY = "waste_to_energy"


class MaterialType(str, Enum):
    """Material types for product composition."""

    PLASTICS = "plastics"
    METALS_FERROUS = "metals_ferrous"
    METALS_ALUMINUM = "metals_aluminum"
    METALS_OTHER = "metals_other"
    PAPER_CARDBOARD = "paper_cardboard"
    GLASS = "glass"
    TEXTILES = "textiles"
    WOOD = "wood"
    RUBBER = "rubber"
    ELECTRONICS = "electronics"
    ORGANIC = "organic"
    CONCRETE = "concrete"
    CERAMICS = "ceramics"
    MIXED = "mixed"
    OTHER = "other"


class ProductCategory(str, Enum):
    """Product categories for sold products."""

    CONSUMER_ELECTRONICS = "consumer_electronics"
    PACKAGING = "packaging"
    CLOTHING_TEXTILES = "clothing_textiles"
    FOOD_BEVERAGE = "food_beverage"
    FURNITURE = "furniture"
    AUTOMOTIVE_PARTS = "automotive_parts"
    BUILDING_MATERIALS = "building_materials"
    INDUSTRIAL_EQUIPMENT = "industrial_equipment"
    HOUSEHOLD_GOODS = "household_goods"
    MEDICAL_DEVICES = "medical_devices"
    TOYS_RECREATION = "toys_recreation"
    OFFICE_SUPPLIES = "office_supplies"
    PERSONAL_CARE = "personal_care"
    BATTERIES = "batteries"
    TIRES = "tires"
    APPLIANCES = "appliances"
    CHEMICALS = "chemicals"
    PHARMACEUTICALS = "pharmaceuticals"
    SOFTWARE_MEDIA = "software_media"
    OTHER = "other"


# ==============================================================================
# BUILT-IN DATA TABLES
# ==============================================================================


# Material x Treatment Emission Factors (kgCO2e per kg of material)
# Sources: EPA WARM v16, DEFRA 2024, IPCC 2006/2019
MATERIAL_TREATMENT_EFS: Dict[str, Dict[str, Decimal]] = {
    "plastics__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "plastics__incineration": {"co2e_per_kg": Decimal("2.33"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "plastics__recycling": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_ferrous__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_ferrous__incineration": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_ferrous__recycling": {"co2e_per_kg": Decimal("0.02"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_aluminum__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_aluminum__incineration": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "metals_aluminum__recycling": {"co2e_per_kg": Decimal("0.03"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "paper_cardboard__landfill": {"co2e_per_kg": Decimal("1.23"), "ch4_per_kg": Decimal("0.046"), "source": "EPA_WARM"},
    "paper_cardboard__incineration": {"co2e_per_kg": Decimal("0.89"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "paper_cardboard__recycling": {"co2e_per_kg": Decimal("0.05"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "paper_cardboard__composting": {"co2e_per_kg": Decimal("0.18"), "ch4_per_kg": Decimal("0.004"), "source": "EPA_WARM"},
    "glass__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "glass__recycling": {"co2e_per_kg": Decimal("0.02"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "textiles__landfill": {"co2e_per_kg": Decimal("1.07"), "ch4_per_kg": Decimal("0.040"), "source": "EPA_WARM"},
    "textiles__incineration": {"co2e_per_kg": Decimal("1.58"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "textiles__recycling": {"co2e_per_kg": Decimal("0.05"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "wood__landfill": {"co2e_per_kg": Decimal("0.82"), "ch4_per_kg": Decimal("0.031"), "source": "EPA_WARM"},
    "wood__incineration": {"co2e_per_kg": Decimal("0.68"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
    "wood__composting": {"co2e_per_kg": Decimal("0.15"), "ch4_per_kg": Decimal("0.003"), "source": "EPA_WARM"},
    "electronics__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "electronics__incineration": {"co2e_per_kg": Decimal("1.44"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "electronics__recycling": {"co2e_per_kg": Decimal("0.03"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "organic__landfill": {"co2e_per_kg": Decimal("1.48"), "ch4_per_kg": Decimal("0.056"), "source": "EPA_WARM"},
    "organic__composting": {"co2e_per_kg": Decimal("0.18"), "ch4_per_kg": Decimal("0.004"), "source": "EPA_WARM"},
    "organic__anaerobic_digestion": {"co2e_per_kg": Decimal("0.10"), "ch4_per_kg": Decimal("0.002"), "source": "EPA_WARM"},
    "rubber__landfill": {"co2e_per_kg": Decimal("0.04"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "rubber__incineration": {"co2e_per_kg": Decimal("2.53"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "concrete__landfill": {"co2e_per_kg": Decimal("0.01"), "ch4_per_kg": Decimal("0.00"), "source": "DEFRA"},
    "mixed__landfill": {"co2e_per_kg": Decimal("0.59"), "ch4_per_kg": Decimal("0.022"), "source": "EPA_WARM"},
    "mixed__incineration": {"co2e_per_kg": Decimal("1.14"), "ch4_per_kg": Decimal("0.00"), "source": "EPA_WARM"},
}

# Avoided emissions factors for recycling (kgCO2e avoided per kg recycled)
RECYCLING_AVOIDED_EFS: Dict[str, Decimal] = {
    "plastics": Decimal("1.02"),
    "metals_ferrous": Decimal("1.81"),
    "metals_aluminum": Decimal("9.13"),
    "paper_cardboard": Decimal("0.87"),
    "glass": Decimal("0.31"),
    "textiles": Decimal("3.10"),
    "electronics": Decimal("2.50"),
    "wood": Decimal("0.45"),
    "rubber": Decimal("0.80"),
}

# Energy recovery credits for waste-to-energy (kgCO2e credit per kg incinerated)
ENERGY_RECOVERY_CREDITS: Dict[str, Decimal] = {
    "plastics": Decimal("0.98"),
    "paper_cardboard": Decimal("0.42"),
    "textiles": Decimal("0.55"),
    "wood": Decimal("0.38"),
    "rubber": Decimal("0.72"),
    "mixed": Decimal("0.46"),
    "organic": Decimal("0.20"),
}


# Default Product Bills of Materials (material fractions summing to 1.0)
DEFAULT_PRODUCT_BOMS: Dict[str, Dict[str, Decimal]] = {
    "consumer_electronics": {
        "plastics": Decimal("0.30"),
        "metals_ferrous": Decimal("0.25"),
        "metals_aluminum": Decimal("0.10"),
        "glass": Decimal("0.15"),
        "electronics": Decimal("0.15"),
        "other": Decimal("0.05"),
    },
    "packaging": {
        "paper_cardboard": Decimal("0.45"),
        "plastics": Decimal("0.35"),
        "glass": Decimal("0.10"),
        "metals_ferrous": Decimal("0.05"),
        "metals_aluminum": Decimal("0.05"),
    },
    "clothing_textiles": {
        "textiles": Decimal("0.85"),
        "plastics": Decimal("0.10"),
        "metals_ferrous": Decimal("0.05"),
    },
    "food_beverage": {
        "organic": Decimal("0.60"),
        "paper_cardboard": Decimal("0.20"),
        "plastics": Decimal("0.15"),
        "glass": Decimal("0.05"),
    },
    "furniture": {
        "wood": Decimal("0.50"),
        "metals_ferrous": Decimal("0.20"),
        "textiles": Decimal("0.15"),
        "plastics": Decimal("0.10"),
        "glass": Decimal("0.05"),
    },
    "automotive_parts": {
        "metals_ferrous": Decimal("0.45"),
        "plastics": Decimal("0.25"),
        "rubber": Decimal("0.15"),
        "metals_aluminum": Decimal("0.10"),
        "glass": Decimal("0.05"),
    },
    "building_materials": {
        "concrete": Decimal("0.40"),
        "metals_ferrous": Decimal("0.25"),
        "wood": Decimal("0.20"),
        "plastics": Decimal("0.10"),
        "glass": Decimal("0.05"),
    },
    "industrial_equipment": {
        "metals_ferrous": Decimal("0.50"),
        "metals_aluminum": Decimal("0.15"),
        "plastics": Decimal("0.15"),
        "rubber": Decimal("0.10"),
        "electronics": Decimal("0.10"),
    },
    "household_goods": {
        "plastics": Decimal("0.35"),
        "metals_ferrous": Decimal("0.20"),
        "glass": Decimal("0.15"),
        "paper_cardboard": Decimal("0.15"),
        "wood": Decimal("0.15"),
    },
    "medical_devices": {
        "plastics": Decimal("0.40"),
        "metals_ferrous": Decimal("0.25"),
        "electronics": Decimal("0.15"),
        "glass": Decimal("0.10"),
        "rubber": Decimal("0.10"),
    },
    "toys_recreation": {
        "plastics": Decimal("0.50"),
        "paper_cardboard": Decimal("0.15"),
        "metals_ferrous": Decimal("0.10"),
        "textiles": Decimal("0.15"),
        "electronics": Decimal("0.10"),
    },
    "office_supplies": {
        "paper_cardboard": Decimal("0.40"),
        "plastics": Decimal("0.30"),
        "metals_ferrous": Decimal("0.15"),
        "wood": Decimal("0.10"),
        "other": Decimal("0.05"),
    },
    "personal_care": {
        "plastics": Decimal("0.45"),
        "glass": Decimal("0.20"),
        "paper_cardboard": Decimal("0.15"),
        "organic": Decimal("0.10"),
        "metals_aluminum": Decimal("0.10"),
    },
    "batteries": {
        "metals_ferrous": Decimal("0.30"),
        "metals_aluminum": Decimal("0.15"),
        "plastics": Decimal("0.20"),
        "electronics": Decimal("0.25"),
        "other": Decimal("0.10"),
    },
    "tires": {
        "rubber": Decimal("0.70"),
        "metals_ferrous": Decimal("0.15"),
        "textiles": Decimal("0.10"),
        "other": Decimal("0.05"),
    },
    "appliances": {
        "metals_ferrous": Decimal("0.40"),
        "plastics": Decimal("0.25"),
        "electronics": Decimal("0.15"),
        "glass": Decimal("0.10"),
        "rubber": Decimal("0.10"),
    },
    "chemicals": {
        "plastics": Decimal("0.50"),
        "glass": Decimal("0.20"),
        "paper_cardboard": Decimal("0.15"),
        "metals_ferrous": Decimal("0.10"),
        "other": Decimal("0.05"),
    },
    "pharmaceuticals": {
        "plastics": Decimal("0.35"),
        "glass": Decimal("0.25"),
        "paper_cardboard": Decimal("0.20"),
        "metals_aluminum": Decimal("0.10"),
        "organic": Decimal("0.10"),
    },
    "software_media": {
        "plastics": Decimal("0.40"),
        "paper_cardboard": Decimal("0.30"),
        "electronics": Decimal("0.20"),
        "metals_ferrous": Decimal("0.10"),
    },
    "other": {
        "mixed": Decimal("1.00"),
    },
}

# Default product weight (kg per unit)
DEFAULT_PRODUCT_WEIGHTS: Dict[str, Decimal] = {
    "consumer_electronics": Decimal("0.50"),
    "packaging": Decimal("0.10"),
    "clothing_textiles": Decimal("0.40"),
    "food_beverage": Decimal("0.30"),
    "furniture": Decimal("15.00"),
    "automotive_parts": Decimal("5.00"),
    "building_materials": Decimal("25.00"),
    "industrial_equipment": Decimal("50.00"),
    "household_goods": Decimal("0.80"),
    "medical_devices": Decimal("0.30"),
    "toys_recreation": Decimal("0.25"),
    "office_supplies": Decimal("0.15"),
    "personal_care": Decimal("0.20"),
    "batteries": Decimal("0.05"),
    "tires": Decimal("10.00"),
    "appliances": Decimal("20.00"),
    "chemicals": Decimal("1.00"),
    "pharmaceuticals": Decimal("0.05"),
    "software_media": Decimal("0.10"),
    "other": Decimal("1.00"),
}


# Regional treatment mixes (fraction of waste going to each pathway)
REGIONAL_TREATMENT_MIXES: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "landfill": Decimal("0.50"),
        "incineration": Decimal("0.12"),
        "recycling": Decimal("0.32"),
        "composting": Decimal("0.06"),
    },
    "EU": {
        "landfill": Decimal("0.24"),
        "incineration": Decimal("0.27"),
        "recycling": Decimal("0.38"),
        "composting": Decimal("0.11"),
    },
    "UK": {
        "landfill": Decimal("0.28"),
        "incineration": Decimal("0.25"),
        "recycling": Decimal("0.35"),
        "composting": Decimal("0.12"),
    },
    "DE": {
        "landfill": Decimal("0.01"),
        "incineration": Decimal("0.32"),
        "recycling": Decimal("0.56"),
        "composting": Decimal("0.11"),
    },
    "JP": {
        "landfill": Decimal("0.05"),
        "incineration": Decimal("0.70"),
        "recycling": Decimal("0.20"),
        "composting": Decimal("0.05"),
    },
    "CN": {
        "landfill": Decimal("0.55"),
        "incineration": Decimal("0.25"),
        "recycling": Decimal("0.17"),
        "composting": Decimal("0.03"),
    },
    "IN": {
        "landfill": Decimal("0.75"),
        "incineration": Decimal("0.05"),
        "recycling": Decimal("0.12"),
        "composting": Decimal("0.08"),
    },
    "BR": {
        "landfill": Decimal("0.60"),
        "incineration": Decimal("0.05"),
        "recycling": Decimal("0.22"),
        "composting": Decimal("0.13"),
    },
    "AU": {
        "landfill": Decimal("0.40"),
        "incineration": Decimal("0.10"),
        "recycling": Decimal("0.42"),
        "composting": Decimal("0.08"),
    },
    "CA": {
        "landfill": Decimal("0.45"),
        "incineration": Decimal("0.04"),
        "recycling": Decimal("0.40"),
        "composting": Decimal("0.11"),
    },
    "KR": {
        "landfill": Decimal("0.15"),
        "incineration": Decimal("0.25"),
        "recycling": Decimal("0.50"),
        "composting": Decimal("0.10"),
    },
    "GLOBAL": {
        "landfill": Decimal("0.40"),
        "incineration": Decimal("0.16"),
        "recycling": Decimal("0.30"),
        "composting": Decimal("0.14"),
    },
}

# Composite EFs for average-data method (kgCO2e per kg of product category)
AVERAGE_DATA_EFS: Dict[str, Decimal] = {
    "consumer_electronics": Decimal("0.82"),
    "packaging": Decimal("0.53"),
    "clothing_textiles": Decimal("0.96"),
    "food_beverage": Decimal("0.61"),
    "furniture": Decimal("0.58"),
    "automotive_parts": Decimal("0.45"),
    "building_materials": Decimal("0.22"),
    "industrial_equipment": Decimal("0.38"),
    "household_goods": Decimal("0.55"),
    "medical_devices": Decimal("0.72"),
    "toys_recreation": Decimal("0.65"),
    "office_supplies": Decimal("0.48"),
    "personal_care": Decimal("0.57"),
    "batteries": Decimal("0.88"),
    "tires": Decimal("1.15"),
    "appliances": Decimal("0.50"),
    "chemicals": Decimal("0.78"),
    "pharmaceuticals": Decimal("0.63"),
    "software_media": Decimal("0.42"),
    "other": Decimal("0.59"),
}


# ==============================================================================
# EndOfLifeTreatmentPipelineEngine
# ==============================================================================


class EndOfLifeTreatmentPipelineEngine:
    """
    EndOfLifeTreatmentPipelineEngine - 10-stage pipeline for Category 12 emissions.

    This engine coordinates the complete end-of-life treatment emissions
    calculation workflow through 10 sequential stages, from input validation
    to sealed audit trail. It supports all treatment pathways (landfill,
    incineration, recycling, composting, anaerobic digestion, open burning,
    wastewater, waste-to-energy).

    Thread Safety:
        Singleton pattern via __new__ with threading.RLock.

    Attributes:
        _compliance_engine: ComplianceCheckerEngine (lazy-loaded)
        _provenance_chains: In-memory provenance chain storage

    Example:
        >>> engine = EndOfLifeTreatmentPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     inputs={"products": [{"name": "Widget", "weight_kg": 0.5}]},
        ...     org_id="org-001",
        ...     year=2025,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    _instance: Optional["EndOfLifeTreatmentPipelineEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "EndOfLifeTreatmentPipelineEngine":
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize EndOfLifeTreatmentPipelineEngine."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._compliance_engine: Optional[Any] = None
        self._provenance_chains: Dict[str, List[Dict[str, Any]]] = {}
        self._pipeline_run_count: int = 0
        self._total_products_processed: int = 0

        self._initialized = True
        logger.info("EndOfLifeTreatmentPipelineEngine initialized (version %s)", ENGINE_VERSION)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None
            logger.info("EndOfLifeTreatmentPipelineEngine singleton reset")

    # ==========================================================================
    # Lazy Engine Loading
    # ==========================================================================

    def _get_compliance_engine(self) -> Optional[Any]:
        """Lazy-load the ComplianceCheckerEngine."""
        if self._compliance_engine is None:
            try:
                from greenlang.end_of_life_treatment.compliance_checker import (
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
        org_id: str = "",
        year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Execute the full 10-stage pipeline for end-of-life treatment emissions.

        Args:
            inputs: Input dictionary containing 'products' list, optional
                'region', 'method', 'treatment_mix', etc.
            org_id: Organization identifier for provenance.
            year: Reporting year.

        Returns:
            Pipeline result dictionary with total_co2e_kg, breakdowns,
            compliance, provenance_hash, and processing metadata.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If pipeline execution fails.
        """
        chain_id = f"eol-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self._provenance_chains[chain_id] = []
        stage_durations: Dict[str, float] = {}

        try:
            # Stage 1: VALIDATE
            start = time.monotonic()
            validated = self._stage_validate(inputs)
            stage_durations["validate"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.VALIDATE, inputs, validated)
            logger.info("[%s] Stage VALIDATE completed in %.2fms", chain_id, stage_durations["validate"])

            # Stage 2: CLASSIFY
            start = time.monotonic()
            classified = self._stage_classify(validated, inputs.get("method"))
            stage_durations["classify"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.CLASSIFY, validated, classified)
            logger.info("[%s] Stage CLASSIFY completed in %.2fms", chain_id, stage_durations["classify"])

            # Stage 3: NORMALIZE
            start = time.monotonic()
            normalized = self._stage_normalize(classified)
            stage_durations["normalize"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.NORMALIZE, classified, normalized)
            logger.info("[%s] Stage NORMALIZE completed in %.2fms", chain_id, stage_durations["normalize"])

            # Stage 4: RESOLVE_EFS
            start = time.monotonic()
            region = inputs.get("region", "GLOBAL")
            ef_resolved = self._stage_resolve_efs(normalized, region)
            stage_durations["resolve_efs"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.RESOLVE_EFS, normalized, ef_resolved)
            logger.info("[%s] Stage RESOLVE_EFS completed in %.2fms", chain_id, stage_durations["resolve_efs"])

            # Stage 5: CALCULATE
            start = time.monotonic()
            calculated = self._stage_calculate(ef_resolved)
            stage_durations["calculate"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.CALCULATE, ef_resolved, calculated)
            logger.info("[%s] Stage CALCULATE completed in %.2fms", chain_id, stage_durations["calculate"])

            # Stage 6: ALLOCATE
            start = time.monotonic()
            allocated = self._stage_allocate(calculated)
            stage_durations["allocate"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.ALLOCATE, calculated, allocated)
            logger.info("[%s] Stage ALLOCATE completed in %.2fms", chain_id, stage_durations["allocate"])

            # Stage 7: AGGREGATE
            start = time.monotonic()
            aggregated = self._stage_aggregate(allocated, year)
            stage_durations["aggregate"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.AGGREGATE, allocated, aggregated)
            logger.info("[%s] Stage AGGREGATE completed in %.2fms", chain_id, stage_durations["aggregate"])

            # Stage 8: COMPLIANCE
            start = time.monotonic()
            compliance = self._stage_compliance(aggregated)
            stage_durations["compliance"] = self._elapsed_ms(start)
            self._record_provenance(chain_id, PipelineStage.COMPLIANCE, aggregated, compliance)
            logger.info("[%s] Stage COMPLIANCE completed in %.2fms", chain_id, stage_durations["compliance"])

            # Stage 9: PROVENANCE
            start = time.monotonic()
            provenance_entries = self._stage_provenance(chain_id, org_id, year)
            stage_durations["provenance"] = self._elapsed_ms(start)
            logger.info("[%s] Stage PROVENANCE completed in %.2fms", chain_id, stage_durations["provenance"])

            # Stage 10: SEAL
            start = time.monotonic()
            provenance_hash = self._stage_seal(chain_id)
            stage_durations["seal"] = self._elapsed_ms(start)
            logger.info("[%s] Stage SEAL completed in %.2fms", chain_id, stage_durations["seal"])

            # Build final result
            total_duration = sum(stage_durations.values())
            self._pipeline_run_count += 1
            self._total_products_processed += len(validated.get("products", []))

            result = {
                "status": PipelineStatus.SUCCESS.value,
                "chain_id": chain_id,
                "org_id": org_id,
                "reporting_year": year,
                "region": region,
                "total_co2e_kg": str(aggregated.get("total_co2e_kg", Decimal("0"))),
                "total_co2e_tonnes": str(
                    (aggregated.get("total_co2e_kg", Decimal("0")) / TONNES_TO_KG).quantize(_QUANT_4DP, rounding=ROUNDING)
                ),
                "by_treatment": aggregated.get("by_treatment", {}),
                "by_material": aggregated.get("by_material", {}),
                "by_category": aggregated.get("by_category", {}),
                "by_region": aggregated.get("by_region", {}),
                "products": allocated.get("products", []),
                "total_weight_kg": str(aggregated.get("total_weight_kg", Decimal("0"))),
                "avoided_emissions_kg": str(aggregated.get("avoided_emissions_kg", Decimal("0"))),
                "energy_recovery_credits_kg": str(aggregated.get("energy_recovery_credits_kg", Decimal("0"))),
                "avoided_reported_separately": True,
                "energy_credits_reported_separately": True,
                "recycling_rate_pct": str(aggregated.get("recycling_rate_pct", Decimal("0"))),
                "diversion_rate_pct": str(aggregated.get("diversion_rate_pct", Decimal("0"))),
                "method": aggregated.get("method", ""),
                "compliance": compliance,
                "provenance_hash": provenance_hash,
                "provenance_entries": len(provenance_entries),
                "processing_time_ms": round(total_duration, 2),
                "stage_durations_ms": stage_durations,
                "product_boundary": "sold_products",
            }

            logger.info(
                "[%s] Pipeline completed in %.2fms. Total: %s kgCO2e",
                chain_id, total_duration, result["total_co2e_kg"],
            )
            return result

        except ValueError:
            raise
        except Exception as e:
            logger.error("[%s] Pipeline execution failed: %s", chain_id, e, exc_info=True)
            raise RuntimeError(f"Pipeline execution failed: {e}") from e
        finally:
            self._provenance_chains.pop(chain_id, None)

    def run_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
        org_id: str = "",
        year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Batch processing for multiple product portfolios.

        Args:
            batch_inputs: List of input dictionaries, each for one run_pipeline call.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Batch result with individual results, totals, and error details.
        """
        start_time = time.monotonic()

        if len(batch_inputs) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(batch_inputs)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_co2e = Decimal("0")

        for idx, inp in enumerate(batch_inputs):
            try:
                r = self.run_pipeline(inp, org_id=org_id, year=year)
                results.append(r)
                total_co2e += Decimal(str(r.get("total_co2e_kg", "0")))
            except Exception as e:
                logger.error("Batch item %d failed: %s", idx, e)
                errors.append({"index": idx, "error": str(e)})

        elapsed = self._elapsed_ms_from(start_time)

        return {
            "status": PipelineStatus.SUCCESS.value if not errors else PipelineStatus.PARTIAL_SUCCESS.value,
            "total_items": len(batch_inputs),
            "successful": len(results),
            "failed": len(errors),
            "total_co2e_kg": str(total_co2e),
            "results": results,
            "errors": errors,
            "processing_time_ms": round(elapsed, 2),
        }

    def run_portfolio_analysis(
        self,
        inputs: Dict[str, Any],
        org_id: str = "",
        year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Portfolio-level analysis with hot-spots, circularity metrics, and material flow.

        Args:
            inputs: Input dictionary with products list.
            org_id: Organization identifier.
            year: Reporting year.

        Returns:
            Portfolio analysis with hot-spots, circularity score, material flow.
        """
        pipeline_result = self.run_pipeline(inputs, org_id=org_id, year=year)

        # Hot-spot analysis (top 5 contributors)
        products = pipeline_result.get("products", [])
        sorted_products = sorted(
            products,
            key=lambda p: Decimal(str(p.get("total_co2e_kg", "0"))),
            reverse=True,
        )
        hot_spots = sorted_products[:5]

        # Circularity score
        recycling_rate = Decimal(str(pipeline_result.get("recycling_rate_pct", "0")))
        diversion_rate = Decimal(str(pipeline_result.get("diversion_rate_pct", "0")))
        circularity_score = (
            (recycling_rate * Decimal("0.6") + diversion_rate * Decimal("0.4"))
        ).quantize(_QUANT_2DP, rounding=ROUNDING)

        # Material flow summary
        by_material = pipeline_result.get("by_material", {})
        total_weight = Decimal(str(pipeline_result.get("total_weight_kg", "0")))
        material_flow = {}
        for mat, data in by_material.items():
            if isinstance(data, dict):
                material_flow[mat] = {
                    "weight_kg": str(data.get("weight_kg", "0")),
                    "co2e_kg": str(data.get("co2e_kg", "0")),
                    "fraction": str(
                        (Decimal(str(data.get("weight_kg", "0"))) / total_weight).quantize(
                            _QUANT_4DP, rounding=ROUNDING
                        ) if total_weight > 0 else Decimal("0")
                    ),
                }

        return {
            **pipeline_result,
            "hot_spots": hot_spots,
            "circularity_score": str(circularity_score),
            "material_flow": material_flow,
            "portfolio_products_count": len(products),
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-flight input validation without running the full pipeline.

        Args:
            inputs: Input dictionary to validate.

        Returns:
            Validation result with is_valid flag and error list.
        """
        errors: List[str] = []

        products = inputs.get("products")
        if not products or not isinstance(products, list):
            errors.append("'products' must be a non-empty list")
        else:
            for idx, prod in enumerate(products):
                if not isinstance(prod, dict):
                    errors.append(f"Product {idx}: must be a dictionary")
                    continue

                name = prod.get("name") or prod.get("product_name")
                if not name:
                    errors.append(f"Product {idx}: 'name' or 'product_name' required")

                weight = prod.get("weight_kg") or prod.get("weight_per_unit_kg")
                units = prod.get("units_sold") or prod.get("quantity")
                total_weight = prod.get("total_weight_kg")
                if weight is None and total_weight is None:
                    errors.append(f"Product {idx}: 'weight_kg' or 'total_weight_kg' required")

                if weight is not None and units is None and total_weight is None:
                    errors.append(f"Product {idx}: 'units_sold' required when 'weight_kg' is per-unit")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "product_count": len(products) if isinstance(products, list) else 0,
        }

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline engine status and metrics.

        Returns:
            Status dictionary with engine info and counters.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "status": "healthy",
            "pipeline_runs": self._pipeline_run_count,
            "total_products_processed": self._total_products_processed,
            "active_chains": len(self._provenance_chains),
        }

    def estimate_runtime(self, n: int) -> Dict[str, Any]:
        """
        Estimate runtime for n products based on historical performance.

        Args:
            n: Number of products to estimate for.

        Returns:
            Runtime estimate dictionary.
        """
        # Baseline: ~2ms per product for in-memory calculation
        base_per_product_ms = Decimal("2.0")
        overhead_ms = Decimal("50.0")

        estimated_ms = overhead_ms + base_per_product_ms * Decimal(str(n))

        return {
            "product_count": n,
            "estimated_ms": str(estimated_ms.quantize(_QUANT_2DP, rounding=ROUNDING)),
            "estimated_seconds": str(
                (estimated_ms / Decimal("1000")).quantize(_QUANT_2DP, rounding=ROUNDING)
            ),
            "note": "Estimate based on in-memory calculation; DB lookups may add latency.",
        }

    # ==========================================================================
    # STAGE 1: VALIDATE
    # ==========================================================================

    def _stage_validate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Input validation.

        Validates:
        - products list is present and non-empty
        - Each product has name, weight, and units sold
        - Material composition fractions sum to ~1.0 if provided
        - Treatment scenario fractions sum to ~1.0 if provided
        """
        errors: List[str] = []

        products = inputs.get("products")
        if not products or not isinstance(products, list):
            raise ValueError("'products' must be a non-empty list")

        validated_products: List[Dict[str, Any]] = []

        for idx, prod in enumerate(products):
            if not isinstance(prod, dict):
                errors.append(f"Product {idx}: must be a dictionary")
                continue

            v_prod: Dict[str, Any] = dict(prod)

            # Name
            name = v_prod.get("name") or v_prod.get("product_name")
            if not name:
                errors.append(f"Product {idx}: 'name' or 'product_name' required")
            v_prod["name"] = name or f"product_{idx}"

            # Weight
            weight_per_unit = v_prod.get("weight_kg") or v_prod.get("weight_per_unit_kg")
            total_weight = v_prod.get("total_weight_kg")
            units_sold = v_prod.get("units_sold") or v_prod.get("quantity")

            if total_weight is not None:
                try:
                    v_prod["total_weight_kg"] = Decimal(str(total_weight))
                except (InvalidOperation, ValueError):
                    errors.append(f"Product {idx}: invalid total_weight_kg")
            elif weight_per_unit is not None and units_sold is not None:
                try:
                    w = Decimal(str(weight_per_unit))
                    u = Decimal(str(units_sold))
                    v_prod["total_weight_kg"] = w * u
                    v_prod["weight_per_unit_kg"] = w
                    v_prod["units_sold"] = int(u)
                except (InvalidOperation, ValueError):
                    errors.append(f"Product {idx}: invalid weight_kg or units_sold")
            else:
                errors.append(
                    f"Product {idx}: 'total_weight_kg' or ('weight_kg' + 'units_sold') required"
                )

            # Material composition validation
            composition = v_prod.get("material_composition")
            if composition and isinstance(composition, dict):
                fraction_sum = sum(Decimal(str(f)) for f in composition.values())
                if abs(fraction_sum - Decimal("1.0")) > Decimal("0.01"):
                    errors.append(
                        f"Product {idx}: material composition fractions sum to "
                        f"{fraction_sum}, expected ~1.0"
                    )

            # Treatment scenario validation
            treatment = v_prod.get("treatment_scenario")
            if treatment and isinstance(treatment, dict):
                t_sum = sum(Decimal(str(f)) for f in treatment.values())
                if abs(t_sum - Decimal("1.0")) > Decimal("0.01"):
                    errors.append(
                        f"Product {idx}: treatment scenario fractions sum to "
                        f"{t_sum}, expected ~1.0"
                    )

            validated_products.append(v_prod)

        if errors:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")

        return {"products": validated_products}

    # ==========================================================================
    # STAGE 2: CLASSIFY
    # ==========================================================================

    def _stage_classify(
        self,
        validated: Dict[str, Any],
        method_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2: Product classification, BOM resolution, treatment pathway ID.

        Assigns:
        - Product category (from name or explicit category field)
        - Material BOM (from product or default table)
        - Calculation method (waste-type-specific / average-data / producer-specific)
        - Treatment pathway (from product or regional default)
        """
        products = validated.get("products", [])
        classified: List[Dict[str, Any]] = []

        for prod in products:
            c_prod = dict(prod)

            # Resolve category
            category = c_prod.get("category") or c_prod.get("product_category")
            if not category:
                category = self._infer_category(c_prod.get("name", ""))
            c_prod["category"] = category

            # Resolve BOM
            composition = c_prod.get("material_composition")
            if not composition or not isinstance(composition, dict):
                default_bom = DEFAULT_PRODUCT_BOMS.get(category, DEFAULT_PRODUCT_BOMS["other"])
                c_prod["material_composition"] = dict(default_bom)
            else:
                # Ensure Decimal fractions
                c_prod["material_composition"] = {
                    k: Decimal(str(v)) for k, v in composition.items()
                }

            # Resolve calculation method
            if method_override:
                c_prod["method"] = method_override
            elif c_prod.get("epd_data") or c_prod.get("producer_ef"):
                c_prod["method"] = CalculationMethod.PRODUCER_SPECIFIC.value
            elif c_prod.get("material_composition"):
                c_prod["method"] = CalculationMethod.WASTE_TYPE_SPECIFIC.value
            else:
                c_prod["method"] = CalculationMethod.AVERAGE_DATA.value

            classified.append(c_prod)

        return {"products": classified}

    # ==========================================================================
    # STAGE 3: NORMALIZE
    # ==========================================================================

    def _stage_normalize(self, classified: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Unit conversions and fraction normalization.

        Converts:
        - lbs -> kg, tonnes -> kg, oz -> kg, short_tons -> kg
        - Normalizes material composition fractions to sum to 1.0
        """
        products = classified.get("products", [])
        normalized: List[Dict[str, Any]] = []

        for prod in products:
            n_prod = dict(prod)

            # Unit conversion for total_weight_kg
            weight_unit = n_prod.get("weight_unit", "kg")
            total_weight = n_prod.get("total_weight_kg", Decimal("0"))
            if isinstance(total_weight, str):
                total_weight = Decimal(total_weight)

            if weight_unit == "lbs":
                total_weight = (total_weight * LBS_TO_KG).quantize(_QUANT_8DP, rounding=ROUNDING)
                n_prod["_original_weight_lbs"] = str(n_prod["total_weight_kg"])
            elif weight_unit in ("tonnes", "t"):
                total_weight = (total_weight * TONNES_TO_KG).quantize(_QUANT_8DP, rounding=ROUNDING)
                n_prod["_original_weight_tonnes"] = str(n_prod["total_weight_kg"])
            elif weight_unit == "oz":
                total_weight = (total_weight * OZ_TO_KG).quantize(_QUANT_8DP, rounding=ROUNDING)
                n_prod["_original_weight_oz"] = str(n_prod["total_weight_kg"])
            elif weight_unit == "short_tons":
                total_weight = (total_weight * SHORT_TONS_TO_KG).quantize(_QUANT_8DP, rounding=ROUNDING)
                n_prod["_original_weight_short_tons"] = str(n_prod["total_weight_kg"])

            n_prod["total_weight_kg"] = total_weight
            n_prod["weight_unit"] = "kg"

            # Normalize material composition fractions
            composition = n_prod.get("material_composition", {})
            if composition:
                frac_sum = sum(Decimal(str(v)) for v in composition.values())
                if frac_sum > Decimal("0") and abs(frac_sum - Decimal("1.0")) > Decimal("0.001"):
                    # Rescale
                    composition = {
                        k: (Decimal(str(v)) / frac_sum).quantize(_QUANT_8DP, rounding=ROUNDING)
                        for k, v in composition.items()
                    }
                n_prod["material_composition"] = composition

            # Default weight if still zero
            if total_weight <= Decimal("0"):
                category = n_prod.get("category", "other")
                default_weight = DEFAULT_PRODUCT_WEIGHTS.get(category, Decimal("1.00"))
                units = n_prod.get("units_sold", 1)
                n_prod["total_weight_kg"] = default_weight * Decimal(str(units))
                n_prod["_weight_defaulted"] = True

            normalized.append(n_prod)

        return {"products": normalized}

    # ==========================================================================
    # STAGE 4: RESOLVE EFS
    # ==========================================================================

    def _stage_resolve_efs(
        self,
        normalized: Dict[str, Any],
        region: str = "GLOBAL",
    ) -> Dict[str, Any]:
        """
        Stage 4: Emission factor resolution.

        Resolves:
        - Material x treatment EFs from built-in table
        - Regional treatment mix if not provided per-product
        - Landfill FOD parameters (implicit in EF table)
        - Incineration parameters (implicit in EF table)
        """
        products = normalized.get("products", [])
        resolved: List[Dict[str, Any]] = []
        treatment_mix = REGIONAL_TREATMENT_MIXES.get(region, REGIONAL_TREATMENT_MIXES["GLOBAL"])

        for prod in products:
            r_prod = dict(prod)

            # Resolve treatment scenario
            treatment = r_prod.get("treatment_scenario")
            if not treatment or not isinstance(treatment, dict):
                r_prod["treatment_scenario"] = dict(treatment_mix)
            else:
                r_prod["treatment_scenario"] = {
                    k: Decimal(str(v)) for k, v in treatment.items()
                }

            # Resolve EFs for each material x treatment
            composition = r_prod.get("material_composition", {})
            treatment_scenario = r_prod["treatment_scenario"]

            ef_map: Dict[str, Dict[str, Any]] = {}
            for material, mat_fraction in composition.items():
                for treatment_key, treat_fraction in treatment_scenario.items():
                    lookup_key = f"{material}__{treatment_key}"
                    ef_entry = MATERIAL_TREATMENT_EFS.get(lookup_key)
                    if ef_entry is not None:
                        ef_map[lookup_key] = {
                            "material": material,
                            "treatment": treatment_key,
                            "co2e_per_kg": ef_entry["co2e_per_kg"],
                            "ch4_per_kg": ef_entry.get("ch4_per_kg", Decimal("0")),
                            "source": ef_entry.get("source", "DEFAULT"),
                            "mat_fraction": Decimal(str(mat_fraction)),
                            "treat_fraction": Decimal(str(treat_fraction)),
                        }
                    else:
                        # Fallback: use mixed EF
                        fallback_key = f"mixed__{treatment_key}"
                        fallback = MATERIAL_TREATMENT_EFS.get(fallback_key)
                        if fallback:
                            ef_map[lookup_key] = {
                                "material": material,
                                "treatment": treatment_key,
                                "co2e_per_kg": fallback["co2e_per_kg"],
                                "ch4_per_kg": fallback.get("ch4_per_kg", Decimal("0")),
                                "source": "FALLBACK_MIXED",
                                "mat_fraction": Decimal(str(mat_fraction)),
                                "treat_fraction": Decimal(str(treat_fraction)),
                            }

            r_prod["ef_map"] = ef_map
            r_prod["region"] = region

            # Resolve average-data EF if using that method
            method = r_prod.get("method", CalculationMethod.WASTE_TYPE_SPECIFIC.value)
            if method == CalculationMethod.AVERAGE_DATA.value:
                category = r_prod.get("category", "other")
                avg_ef = AVERAGE_DATA_EFS.get(category, AVERAGE_DATA_EFS["other"])
                r_prod["average_data_ef"] = avg_ef

            # Resolve producer-specific EF if provided
            if method == CalculationMethod.PRODUCER_SPECIFIC.value:
                epd_ef = r_prod.get("producer_ef") or r_prod.get("epd_data", {}).get("eol_co2e_per_kg")
                if epd_ef is not None:
                    r_prod["producer_ef"] = Decimal(str(epd_ef))

            resolved.append(r_prod)

        return {"products": resolved, "region": region}

    # ==========================================================================
    # STAGE 5: CALCULATE
    # ==========================================================================

    def _stage_calculate(self, ef_resolved: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: Emissions calculation. Method dispatch per product.

        Methods:
        - waste_type_specific: material_weight * mat_fraction * treat_fraction * EF
        - average_data: total_weight * average_data_EF
        - producer_specific: total_weight * producer_ef
        """
        products = ef_resolved.get("products", [])
        calculated: List[Dict[str, Any]] = []

        for prod in products:
            c_prod = dict(prod)
            method = c_prod.get("method", CalculationMethod.WASTE_TYPE_SPECIFIC.value)
            total_weight = Decimal(str(c_prod.get("total_weight_kg", "0")))

            if method == CalculationMethod.WASTE_TYPE_SPECIFIC.value:
                result = self._calc_waste_type_specific(c_prod, total_weight)
            elif method == CalculationMethod.AVERAGE_DATA.value:
                result = self._calc_average_data(c_prod, total_weight)
            elif method == CalculationMethod.PRODUCER_SPECIFIC.value:
                result = self._calc_producer_specific(c_prod, total_weight)
            else:
                result = self._calc_waste_type_specific(c_prod, total_weight)

            c_prod.update(result)
            calculated.append(c_prod)

        return {"products": calculated}

    def _calc_waste_type_specific(
        self, prod: Dict[str, Any], total_weight: Decimal
    ) -> Dict[str, Any]:
        """Calculate using waste-type-specific method."""
        ef_map = prod.get("ef_map", {})
        total_co2e = Decimal("0")
        total_ch4 = Decimal("0")
        by_material: Dict[str, Decimal] = {}
        by_treatment: Dict[str, Decimal] = {}
        avoided = Decimal("0")
        energy_credits = Decimal("0")

        for key, ef_data in ef_map.items():
            material = ef_data["material"]
            treatment = ef_data["treatment"]
            mat_frac = ef_data["mat_fraction"]
            treat_frac = ef_data["treat_fraction"]
            co2e_per_kg = ef_data["co2e_per_kg"]
            ch4_per_kg = ef_data.get("ch4_per_kg", Decimal("0"))

            weight_for_combo = (total_weight * mat_frac * treat_frac).quantize(
                _QUANT_8DP, rounding=ROUNDING
            )

            co2e = (weight_for_combo * co2e_per_kg).quantize(_QUANT_8DP, rounding=ROUNDING)
            ch4 = (weight_for_combo * ch4_per_kg).quantize(_QUANT_8DP, rounding=ROUNDING)

            total_co2e += co2e
            total_ch4 += ch4

            by_material[material] = by_material.get(material, Decimal("0")) + co2e
            by_treatment[treatment] = by_treatment.get(treatment, Decimal("0")) + co2e

            # Avoided emissions from recycling
            if treatment == "recycling":
                avoided_ef = RECYCLING_AVOIDED_EFS.get(material, Decimal("0"))
                avoided += (weight_for_combo * avoided_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

            # Energy recovery credits
            if treatment in ("incineration", "waste_to_energy"):
                credit_ef = ENERGY_RECOVERY_CREDITS.get(material, Decimal("0"))
                energy_credits += (weight_for_combo * credit_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

        return {
            "total_co2e_kg": total_co2e.quantize(_QUANT_8DP, rounding=ROUNDING),
            "ch4_kg": total_ch4.quantize(_QUANT_8DP, rounding=ROUNDING),
            "by_material_co2e": {k: str(v) for k, v in by_material.items()},
            "by_treatment_co2e": {k: str(v) for k, v in by_treatment.items()},
            "avoided_emissions_kg": avoided.quantize(_QUANT_8DP, rounding=ROUNDING),
            "energy_recovery_credits_kg": energy_credits.quantize(_QUANT_8DP, rounding=ROUNDING),
            "calculation_method": CalculationMethod.WASTE_TYPE_SPECIFIC.value,
        }

    def _calc_average_data(
        self, prod: Dict[str, Any], total_weight: Decimal
    ) -> Dict[str, Any]:
        """Calculate using average-data method."""
        avg_ef = prod.get("average_data_ef", AVERAGE_DATA_EFS.get("other", Decimal("0.59")))
        if isinstance(avg_ef, str):
            avg_ef = Decimal(avg_ef)

        total_co2e = (total_weight * avg_ef).quantize(_QUANT_8DP, rounding=ROUNDING)
        category = prod.get("category", "other")

        return {
            "total_co2e_kg": total_co2e,
            "ch4_kg": Decimal("0"),
            "by_material_co2e": {},
            "by_treatment_co2e": {category: str(total_co2e)},
            "avoided_emissions_kg": Decimal("0"),
            "energy_recovery_credits_kg": Decimal("0"),
            "calculation_method": CalculationMethod.AVERAGE_DATA.value,
        }

    def _calc_producer_specific(
        self, prod: Dict[str, Any], total_weight: Decimal
    ) -> Dict[str, Any]:
        """Calculate using producer-specific (EPD) method."""
        producer_ef = prod.get("producer_ef", Decimal("0.50"))
        if isinstance(producer_ef, str):
            producer_ef = Decimal(producer_ef)

        total_co2e = (total_weight * producer_ef).quantize(_QUANT_8DP, rounding=ROUNDING)

        # EPD may provide separate avoided emissions
        epd_avoided = prod.get("epd_data", {}).get("avoided_co2e_per_kg", Decimal("0"))
        if isinstance(epd_avoided, str):
            epd_avoided = Decimal(epd_avoided)
        avoided = (total_weight * epd_avoided).quantize(_QUANT_8DP, rounding=ROUNDING)

        return {
            "total_co2e_kg": total_co2e,
            "ch4_kg": Decimal("0"),
            "by_material_co2e": {},
            "by_treatment_co2e": {"producer_specific": str(total_co2e)},
            "avoided_emissions_kg": avoided,
            "energy_recovery_credits_kg": Decimal("0"),
            "calculation_method": CalculationMethod.PRODUCER_SPECIFIC.value,
        }

    # ==========================================================================
    # STAGE 6: ALLOCATE
    # ==========================================================================

    def _stage_allocate(self, calculated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 6: Proportional allocation across products, materials, treatments.

        Computes:
        - Per-product contribution percentages
        - Material stream weights
        - Treatment method totals
        """
        products = calculated.get("products", [])
        total_co2e = sum(
            Decimal(str(p.get("total_co2e_kg", "0"))) for p in products
        )

        allocated_products: List[Dict[str, Any]] = []
        for prod in products:
            a_prod = dict(prod)
            prod_co2e = Decimal(str(prod.get("total_co2e_kg", "0")))

            if total_co2e > Decimal("0"):
                contribution_pct = (prod_co2e / total_co2e * Decimal("100")).quantize(
                    _QUANT_2DP, rounding=ROUNDING
                )
            else:
                contribution_pct = Decimal("0")

            a_prod["contribution_pct"] = str(contribution_pct)
            allocated_products.append(a_prod)

        return {"products": allocated_products, "total_co2e_kg": total_co2e}

    # ==========================================================================
    # STAGE 7: AGGREGATE
    # ==========================================================================

    def _stage_aggregate(
        self, allocated: Dict[str, Any], year: int
    ) -> Dict[str, Any]:
        """
        Stage 7: Multi-dimensional aggregation.

        Aggregates by:
        - treatment method
        - material type
        - product category
        - region
        - period (year)
        """
        products = allocated.get("products", [])

        total_co2e = Decimal("0")
        total_weight = Decimal("0")
        total_avoided = Decimal("0")
        total_energy_credits = Decimal("0")
        by_treatment: Dict[str, Dict[str, Any]] = {}
        by_material: Dict[str, Dict[str, Any]] = {}
        by_category: Dict[str, Dict[str, Any]] = {}
        by_region: Dict[str, Dict[str, Any]] = {}
        methods_used: set = set()

        recycled_weight = Decimal("0")
        landfilled_weight = Decimal("0")

        for prod in products:
            prod_co2e = Decimal(str(prod.get("total_co2e_kg", "0")))
            prod_weight = Decimal(str(prod.get("total_weight_kg", "0")))
            prod_avoided = Decimal(str(prod.get("avoided_emissions_kg", "0")))
            prod_energy = Decimal(str(prod.get("energy_recovery_credits_kg", "0")))

            total_co2e += prod_co2e
            total_weight += prod_weight
            total_avoided += prod_avoided
            total_energy_credits += prod_energy
            methods_used.add(prod.get("calculation_method", "unknown"))

            # By treatment
            by_treat = prod.get("by_treatment_co2e", {})
            for treat, co2e_str in by_treat.items():
                co2e = Decimal(str(co2e_str))
                if treat not in by_treatment:
                    by_treatment[treat] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0")}
                by_treatment[treat]["co2e_kg"] += co2e

            # By material
            by_mat = prod.get("by_material_co2e", {})
            for mat, co2e_str in by_mat.items():
                co2e = Decimal(str(co2e_str))
                if mat not in by_material:
                    by_material[mat] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0")}
                by_material[mat]["co2e_kg"] += co2e

            # Compute material weights
            composition = prod.get("material_composition", {})
            for mat, frac in composition.items():
                frac_d = Decimal(str(frac))
                mat_weight = (prod_weight * frac_d).quantize(_QUANT_8DP, rounding=ROUNDING)
                if mat not in by_material:
                    by_material[mat] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0")}
                by_material[mat]["weight_kg"] += mat_weight

            # Treatment weights
            treatment_scenario = prod.get("treatment_scenario", {})
            for treat, frac in treatment_scenario.items():
                frac_d = Decimal(str(frac))
                treat_weight = (prod_weight * frac_d).quantize(_QUANT_8DP, rounding=ROUNDING)
                if treat not in by_treatment:
                    by_treatment[treat] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0")}
                by_treatment[treat]["weight_kg"] += treat_weight

                if treat == "recycling":
                    recycled_weight += treat_weight
                elif treat == "landfill":
                    landfilled_weight += treat_weight

            # By category
            category = prod.get("category", "other")
            if category not in by_category:
                by_category[category] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0"), "count": 0}
            by_category[category]["co2e_kg"] += prod_co2e
            by_category[category]["weight_kg"] += prod_weight
            by_category[category]["count"] += 1

            # By region
            region = prod.get("region", "GLOBAL")
            if region not in by_region:
                by_region[region] = {"co2e_kg": Decimal("0"), "weight_kg": Decimal("0")}
            by_region[region]["co2e_kg"] += prod_co2e
            by_region[region]["weight_kg"] += prod_weight

        # Serialize Decimal values
        for d in [by_treatment, by_material, by_category, by_region]:
            for key in d:
                for sub_key in d[key]:
                    if isinstance(d[key][sub_key], Decimal):
                        d[key][sub_key] = str(d[key][sub_key])

        # Circularity metrics
        recycling_rate = Decimal("0")
        diversion_rate = Decimal("0")
        if total_weight > Decimal("0"):
            recycling_rate = (recycled_weight / total_weight * Decimal("100")).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )
            diverted = total_weight - landfilled_weight
            diversion_rate = (diverted / total_weight * Decimal("100")).quantize(
                _QUANT_2DP, rounding=ROUNDING
            )

        # Determine overall method
        if len(methods_used) == 1:
            method = methods_used.pop()
        elif len(methods_used) > 1:
            method = CalculationMethod.HYBRID.value
        else:
            method = "unknown"

        return {
            "total_co2e_kg": total_co2e,
            "total_weight_kg": total_weight,
            "avoided_emissions_kg": total_avoided,
            "energy_recovery_credits_kg": total_energy_credits,
            "by_treatment": by_treatment,
            "by_material": by_material,
            "by_category": by_category,
            "by_region": by_region,
            "recycling_rate_pct": recycling_rate,
            "diversion_rate_pct": diversion_rate,
            "reporting_year": year,
            "method": method,
        }

    # ==========================================================================
    # STAGE 8: COMPLIANCE
    # ==========================================================================

    def _stage_compliance(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 8: Delegate to ComplianceCheckerEngine."""
        engine = self._get_compliance_engine()
        if engine is None:
            return {"status": "skipped", "reason": "ComplianceCheckerEngine not available"}

        try:
            # Build compliance input
            compliance_input = {
                "total_co2e": str(aggregated.get("total_co2e_kg", Decimal("0"))),
                "total_co2e_kg": str(aggregated.get("total_co2e_kg", Decimal("0"))),
                "treatment_breakdown": aggregated.get("by_treatment", {}),
                "material_breakdown": aggregated.get("by_material", {}),
                "by_category": aggregated.get("by_category", {}),
                "method": aggregated.get("method", ""),
                "calculation_method": aggregated.get("method", ""),
                "reporting_year": aggregated.get("reporting_year"),
                "recycling_rate_pct": str(aggregated.get("recycling_rate_pct", "0")),
                "diversion_rate_pct": str(aggregated.get("diversion_rate_pct", "0")),
                "avoided_emissions": str(aggregated.get("avoided_emissions_kg", "0")),
                "avoided_reported_separately": True,
                "energy_recovery_credits": str(aggregated.get("energy_recovery_credits_kg", "0")),
                "energy_credits_reported_separately": True,
                "product_boundary": "sold_products",
            }

            results = engine.check_all_frameworks(compliance_input)
            summary = engine.get_compliance_summary(results)
            return summary

        except Exception as e:
            logger.error("Compliance check failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    # ==========================================================================
    # STAGE 9: PROVENANCE
    # ==========================================================================

    def _stage_provenance(
        self, chain_id: str, org_id: str, year: int
    ) -> List[Dict[str, Any]]:
        """Stage 9: Build provenance chain entries."""
        chain = self._provenance_chains.get(chain_id, [])

        # Add metadata entry
        chain.append({
            "stage": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "org_id": org_id,
            "reporting_year": year,
            "agent_id": "GL-MRV-S3-012",
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
        })

        self._provenance_chains[chain_id] = chain
        return chain

    # ==========================================================================
    # STAGE 10: SEAL
    # ==========================================================================

    def _stage_seal(self, chain_id: str) -> str:
        """Stage 10: SHA-256 hash of complete provenance chain."""
        chain = self._provenance_chains.get(chain_id, [])

        # Serialize chain for hashing
        chain_str = json.dumps(chain, sort_keys=True, default=str)
        provenance_hash = hashlib.sha256(chain_str.encode("utf-8")).hexdigest()

        logger.info(
            "[%s] Provenance sealed: %d entries, hash=%s",
            chain_id,
            len(chain),
            provenance_hash[:16],
        )

        return provenance_hash

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _record_provenance(
        self,
        chain_id: str,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
    ) -> None:
        """Record a provenance entry for a pipeline stage."""
        chain = self._provenance_chains.get(chain_id, [])

        # Create a compact representation for hashing
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

        output_hash = hashlib.sha256(
            json.dumps(output_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]

        chain.append({
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": input_hash,
            "output_hash": output_hash,
        })

        self._provenance_chains[chain_id] = chain

    @staticmethod
    def _infer_category(product_name: str) -> str:
        """Infer product category from name using keyword matching."""
        name_lower = product_name.lower()

        category_keywords: Dict[str, List[str]] = {
            "consumer_electronics": ["phone", "laptop", "tablet", "computer", "monitor", "tv", "electronic"],
            "packaging": ["box", "carton", "bag", "wrapper", "packaging", "container"],
            "clothing_textiles": ["shirt", "pants", "dress", "clothing", "textile", "fabric", "apparel"],
            "food_beverage": ["food", "beverage", "drink", "snack", "meal", "grocery"],
            "furniture": ["chair", "table", "desk", "sofa", "bed", "furniture", "shelf"],
            "automotive_parts": ["auto", "car", "vehicle", "brake", "engine", "tire"],
            "building_materials": ["brick", "cement", "steel", "lumber", "insulation", "building"],
            "industrial_equipment": ["machine", "pump", "motor", "compressor", "industrial"],
            "household_goods": ["pan", "pot", "dish", "utensil", "household", "kitchen"],
            "medical_devices": ["medical", "surgical", "implant", "diagnostic", "syringe"],
            "toys_recreation": ["toy", "game", "sport", "recreation", "play"],
            "office_supplies": ["pen", "paper", "stapler", "office", "binder", "folder"],
            "personal_care": ["soap", "shampoo", "cosmetic", "lotion", "personal care"],
            "batteries": ["battery", "cell", "lithium", "alkaline"],
            "tires": ["tire", "tyre", "wheel"],
            "appliances": ["washer", "dryer", "refrigerator", "oven", "appliance", "dishwasher"],
            "chemicals": ["chemical", "solvent", "adhesive", "paint", "coating"],
            "pharmaceuticals": ["drug", "medicine", "pharma", "pill", "capsule"],
            "software_media": ["software", "cd", "dvd", "disc", "media", "book"],
        }

        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category

        return "other"

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Calculate elapsed milliseconds from monotonic start time."""
        return (time.monotonic() - start) * 1000.0

    @staticmethod
    def _elapsed_ms_from(start: float) -> float:
        """Calculate elapsed milliseconds from monotonic start time."""
        return (time.monotonic() - start) * 1000.0


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "ENGINE_ID",
    "ENGINE_VERSION",
    "PipelineStage",
    "PipelineStatus",
    "CalculationMethod",
    "TreatmentPathway",
    "MaterialType",
    "ProductCategory",
    "MATERIAL_TREATMENT_EFS",
    "RECYCLING_AVOIDED_EFS",
    "ENERGY_RECOVERY_CREDITS",
    "DEFAULT_PRODUCT_BOMS",
    "DEFAULT_PRODUCT_WEIGHTS",
    "REGIONAL_TREATMENT_MIXES",
    "AVERAGE_DATA_EFS",
    "EndOfLifeTreatmentPipelineEngine",
]
