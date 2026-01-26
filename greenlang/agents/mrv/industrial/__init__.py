# -*- coding: utf-8 -*-
"""
GreenLang Industrial Sector MRV Agents
======================================

This package contains MRV (Measurement, Reporting, Verification) agents
for industrial sectors:

Agents:
    - GL-MRV-IND-001: Steel Production MRV
    - GL-MRV-IND-002: Cement Production MRV
    - GL-MRV-IND-003: Chemicals Production MRV
    - GL-MRV-IND-004: Aluminum Production MRV
    - GL-MRV-IND-005: Pulp & Paper MRV
    - GL-MRV-IND-006: Glass Production MRV
    - GL-MRV-IND-007: Food Processing MRV
    - GL-MRV-IND-008: Pharmaceutical MRV
    - GL-MRV-IND-009: Electronics MRV
    - GL-MRV-IND-010: Automotive MRV
    - GL-MRV-IND-011: Textiles MRV
    - GL-MRV-IND-012: Mining MRV
    - GL-MRV-IND-013: Plastics MRV

All agents implement:
    - Zero-hallucination calculations
    - CBAM compliance (EU 2023/956)
    - SHA-256 provenance tracking
    - GHG Protocol alignment

Author: GreenLang Framework Team
Version: 1.0.0
"""

from typing import List

# Base classes
from .base import (
    IndustrialMRVBaseAgent,
    IndustrialMRVInput,
    IndustrialMRVOutput,
    CalculationStep,
    EmissionFactor,
    CBAMOutput,
    EmissionScope,
    VerificationStatus,
    DataQuality,
)

# GL-MRV-IND-001: Steel
from .steel_mrv import (
    SteelProductionMRVAgent,
    SteelMRVInput,
    SteelMRVOutput,
    SteelProductionRoute,
    HydrogenSource,
)

# GL-MRV-IND-002: Cement
from .cement_mrv import (
    CementProductionMRVAgent,
    CementMRVInput,
    CementMRVOutput,
    CementType,
    KilnFuelType,
    SCMType,
)

# GL-MRV-IND-003: Chemicals
from .chemicals_mrv import (
    ChemicalsProductionMRVAgent,
    ChemicalsMRVInput,
    ChemicalsMRVOutput,
    ChemicalProduct,
    FeedstockType,
)

# GL-MRV-IND-004: Aluminum
from .aluminum_mrv import (
    AluminumProductionMRVAgent,
    AluminumMRVInput,
    AluminumMRVOutput,
    AluminumProductionRoute,
    AnodeTechnology,
)

# GL-MRV-IND-005: Pulp & Paper
from .pulp_paper_mrv import (
    PulpPaperMRVAgent,
    PulpPaperMRVInput,
    PulpPaperMRVOutput,
    PulpType,
    PaperType,
)

# GL-MRV-IND-006: Glass
from .glass_mrv import (
    GlassProductionMRVAgent,
    GlassMRVInput,
    GlassMRVOutput,
    GlassType,
    FurnaceType,
)

# GL-MRV-IND-007: Food Processing
from .food_processing_mrv import (
    FoodProcessingMRVAgent,
    FoodProcessingMRVInput,
    FoodProcessingMRVOutput,
    FoodSubsector,
)

# GL-MRV-IND-008 to IND-013: Additional sectors
from .additional_sectors import (
    # Pharmaceutical
    PharmaceuticalMRVAgent,
    PharmaMRVInput,
    PharmaMRVOutput,
    PharmaProcessType,
    # Electronics
    ElectronicsMRVAgent,
    ElectronicsMRVInput,
    ElectronicsMRVOutput,
    ElectronicsProductType,
    # Automotive
    AutomotiveMRVAgent,
    AutomotiveMRVInput,
    AutomotiveMRVOutput,
    AutomotiveProcessType,
    # Textiles
    TextilesMRVAgent,
    TextilesMRVInput,
    TextilesMRVOutput,
    TextileProcessType,
    # Mining
    MiningMRVAgent,
    MiningMRVInput,
    MiningMRVOutput,
    MiningType,
    # Plastics
    PlasticsMRVAgent,
    PlasticsMRVInput,
    PlasticsMRVOutput,
    PlasticType,
)

__all__: List[str] = [
    # Base
    "IndustrialMRVBaseAgent",
    "IndustrialMRVInput",
    "IndustrialMRVOutput",
    "CalculationStep",
    "EmissionFactor",
    "CBAMOutput",
    "EmissionScope",
    "VerificationStatus",
    "DataQuality",
    # Steel
    "SteelProductionMRVAgent",
    "SteelMRVInput",
    "SteelMRVOutput",
    "SteelProductionRoute",
    "HydrogenSource",
    # Cement
    "CementProductionMRVAgent",
    "CementMRVInput",
    "CementMRVOutput",
    "CementType",
    "KilnFuelType",
    "SCMType",
    # Chemicals
    "ChemicalsProductionMRVAgent",
    "ChemicalsMRVInput",
    "ChemicalsMRVOutput",
    "ChemicalProduct",
    "FeedstockType",
    # Aluminum
    "AluminumProductionMRVAgent",
    "AluminumMRVInput",
    "AluminumMRVOutput",
    "AluminumProductionRoute",
    "AnodeTechnology",
    # Pulp & Paper
    "PulpPaperMRVAgent",
    "PulpPaperMRVInput",
    "PulpPaperMRVOutput",
    "PulpType",
    "PaperType",
    # Glass
    "GlassProductionMRVAgent",
    "GlassMRVInput",
    "GlassMRVOutput",
    "GlassType",
    "FurnaceType",
    # Food Processing
    "FoodProcessingMRVAgent",
    "FoodProcessingMRVInput",
    "FoodProcessingMRVOutput",
    "FoodSubsector",
    # Pharmaceutical
    "PharmaceuticalMRVAgent",
    "PharmaMRVInput",
    "PharmaMRVOutput",
    "PharmaProcessType",
    # Electronics
    "ElectronicsMRVAgent",
    "ElectronicsMRVInput",
    "ElectronicsMRVOutput",
    "ElectronicsProductType",
    # Automotive
    "AutomotiveMRVAgent",
    "AutomotiveMRVInput",
    "AutomotiveMRVOutput",
    "AutomotiveProcessType",
    # Textiles
    "TextilesMRVAgent",
    "TextilesMRVInput",
    "TextilesMRVOutput",
    "TextileProcessType",
    # Mining
    "MiningMRVAgent",
    "MiningMRVInput",
    "MiningMRVOutput",
    "MiningType",
    # Plastics
    "PlasticsMRVAgent",
    "PlasticsMRVInput",
    "PlasticsMRVOutput",
    "PlasticType",
]

__version__ = "1.0.0"
