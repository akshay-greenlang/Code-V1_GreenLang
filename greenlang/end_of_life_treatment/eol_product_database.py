# -*- coding: utf-8 -*-
"""
EOLProductDatabaseEngine - End-of-Life Treatment emission factor database engine.

This module implements the EOLProductDatabaseEngine for AGENT-MRV-025
(GL-MRV-S3-012) End-of-Life Treatment of Sold Products. It provides thread-safe
singleton access to emission factor databases, product composition tables (BOM),
regional treatment mix profiles, IPCC FOD landfill parameters, incineration
parameters, recycling/avoided emission credits, composting/AD factors, and
product weight defaults.

Key Differentiator from Category 5 (Waste Generated in Operations):
    Category 5 covers waste from the reporting company's own operations.
    Category 12 covers end-of-life treatment of products SOLD by the company,
    disposed of by downstream consumers and third parties.

Features:
- 15 material types with treatment-specific EFs (kgCO2e/kg)
- 20 product categories with default material compositions (BOM)
- 12 regional treatment mix profiles (US, EU, JP, CN, IN, BR, UK, AU, KR, CA, MX, GLOBAL)
- IPCC FOD parameters by material and climate zone (DOC, DOCf, MCF, k, F, OX)
- Incineration parameters (dry_matter, carbon_fraction, fossil_carbon, oxidation)
- Gas collection efficiency by landfill type (0.0 - 0.75)
- Energy recovery factors by region (WtE efficiency, displaced grid EF)
- Recycling processing EFs (transport + MRF energy per material)
- Avoided emission credits (virgin material substitution, negative values)
- Product weight defaults (average kg per unit for 20 categories)
- Composting and anaerobic digestion emission factors
- Thread-safe singleton pattern with threading.RLock()
- Zero-hallucination factor retrieval (no LLM in lookup path)
- Performance timing via time.monotonic()
- Provenance tracking via SHA-256 hashes

Reference Sources:
    - GHG Protocol Scope 3 Standard, Category 12 Technical Guidance
    - IPCC 2006 Guidelines Vol 5 (Waste Sector)
    - IPCC 2019 Refinement to the 2006 Guidelines
    - EPA WARM v16 (Waste Reduction Model)
    - DEFRA/DESNZ 2024 Conversion Factors
    - IEA CO2 Emissions from Fuel Combustion (grid EFs)

Example:
    >>> engine = EOLProductDatabaseEngine()
    >>> ef = engine.get_material_ef("plastic", "incineration")
    >>> ef
    Decimal('2.760')
    >>> comp = engine.get_product_composition("electronics")
    >>> comp
    {'plastic': Decimal('0.40'), 'metal': Decimal('0.35'), ...}
    >>> mix = engine.get_regional_treatment_mix("US")
    >>> mix
    {'landfill': Decimal('0.50'), 'incineration': Decimal('0.14'), ...}

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
ENGINE_ID: str = "eol_product_database_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"

# ==============================================================================
# DECIMAL PRECISION
# ==============================================================================

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANT_8DP = Decimal("0.00000001")
_QUANT_6DP = Decimal("0.000001")

# ==============================================================================
# VALID ENUMERATIONS
# ==============================================================================

VALID_MATERIALS: Set[str] = {
    "plastic", "metal", "aluminum", "steel", "glass", "paper",
    "cardboard", "wood", "textile", "electronics", "organic",
    "rubber", "ceramic", "concrete", "mixed",
}

VALID_TREATMENTS: Set[str] = {
    "landfill", "incineration", "recycling", "composting",
    "anaerobic_digestion", "open_burning",
}

VALID_PRODUCT_CATEGORIES: Set[str] = {
    "electronics", "appliances", "packaging", "clothing", "footwear",
    "furniture", "automotive_parts", "batteries", "tires", "toys",
    "food_products", "beverages", "personal_care", "cleaning_products",
    "building_materials", "paper_products", "glass_containers",
    "metal_containers", "medical_devices", "general_consumer_goods",
}

VALID_REGIONS: Set[str] = {
    "US", "EU", "JP", "CN", "IN", "BR", "UK", "AU", "KR", "CA", "MX", "GLOBAL",
}

VALID_CLIMATE_ZONES: Set[str] = {
    "boreal_temperate_dry", "temperate_wet", "tropical_dry", "tropical_wet",
}

VALID_LANDFILL_TYPES: Set[str] = {
    "unmanaged", "managed_anaerobic", "managed_semi_aerobic",
    "unmanaged_deep", "unmanaged_shallow", "engineered_with_gas",
}


# ==============================================================================
# MATERIAL EMISSION FACTORS DATABASE
# ==============================================================================
# Treatment-specific EFs in kgCO2e/kg of material.
# Sources: EPA WARM v16, DEFRA 2024, IPCC 2006/2019
# Negative values for recycling represent avoided emissions from virgin
# material displacement (cut-off approach: reported separately).

MATERIAL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "plastic": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("2.760"),
        "recycling": Decimal("-1.440"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("3.100"),
    },
    "metal": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-4.200"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "aluminum": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-9.120"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "steel": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-1.820"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "glass": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.314"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "paper": {
        "landfill": Decimal("1.095"),
        "incineration": Decimal("1.320"),
        "recycling": Decimal("-0.840"),
        "composting": Decimal("0.176"),
        "anaerobic_digestion": Decimal("0.088"),
        "open_burning": Decimal("1.500"),
    },
    "cardboard": {
        "landfill": Decimal("0.875"),
        "incineration": Decimal("1.100"),
        "recycling": Decimal("-0.720"),
        "composting": Decimal("0.155"),
        "anaerobic_digestion": Decimal("0.077"),
        "open_burning": Decimal("1.250"),
    },
    "wood": {
        "landfill": Decimal("0.645"),
        "incineration": Decimal("1.410"),
        "recycling": Decimal("-0.580"),
        "composting": Decimal("0.132"),
        "anaerobic_digestion": Decimal("0.066"),
        "open_burning": Decimal("1.600"),
    },
    "textile": {
        "landfill": Decimal("0.450"),
        "incineration": Decimal("2.330"),
        "recycling": Decimal("-3.100"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("2.600"),
    },
    "electronics": {
        "landfill": Decimal("0.042"),
        "incineration": Decimal("1.180"),
        "recycling": Decimal("-2.950"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("1.400"),
    },
    "organic": {
        "landfill": Decimal("1.824"),
        "incineration": Decimal("0.580"),
        "recycling": Decimal("0.000"),
        "composting": Decimal("0.176"),
        "anaerobic_digestion": Decimal("0.088"),
        "open_burning": Decimal("0.700"),
    },
    "rubber": {
        "landfill": Decimal("0.042"),
        "incineration": Decimal("2.850"),
        "recycling": Decimal("-1.220"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("3.200"),
    },
    "ceramic": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.150"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "concrete": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.024"),
        "composting": Decimal("0.000"),
        "anaerobic_digestion": Decimal("0.000"),
        "open_burning": Decimal("0.021"),
    },
    "mixed": {
        "landfill": Decimal("0.587"),
        "incineration": Decimal("1.350"),
        "recycling": Decimal("-0.680"),
        "composting": Decimal("0.088"),
        "anaerobic_digestion": Decimal("0.044"),
        "open_burning": Decimal("1.550"),
    },
}


# ==============================================================================
# PRODUCT CATEGORY COMPOSITIONS (BILL OF MATERIALS)
# ==============================================================================
# Default material composition fractions for 20 product categories.
# Fractions must sum to 1.0 for each product.
# Sources: EPA product characterization studies, industry average BOMs

PRODUCT_COMPOSITIONS: Dict[str, Dict[str, Decimal]] = {
    "electronics": {
        "plastic": Decimal("0.40"),
        "metal": Decimal("0.35"),
        "glass": Decimal("0.15"),
        "mixed": Decimal("0.10"),
    },
    "appliances": {
        "metal": Decimal("0.60"),
        "plastic": Decimal("0.30"),
        "mixed": Decimal("0.10"),
    },
    "packaging": {
        "paper": Decimal("0.50"),
        "plastic": Decimal("0.30"),
        "mixed": Decimal("0.20"),
    },
    "clothing": {
        "textile": Decimal("0.90"),
        "mixed": Decimal("0.10"),
    },
    "footwear": {
        "rubber": Decimal("0.40"),
        "textile": Decimal("0.35"),
        "plastic": Decimal("0.15"),
        "mixed": Decimal("0.10"),
    },
    "furniture": {
        "wood": Decimal("0.55"),
        "metal": Decimal("0.20"),
        "textile": Decimal("0.15"),
        "mixed": Decimal("0.10"),
    },
    "automotive_parts": {
        "steel": Decimal("0.45"),
        "aluminum": Decimal("0.20"),
        "plastic": Decimal("0.20"),
        "rubber": Decimal("0.10"),
        "mixed": Decimal("0.05"),
    },
    "batteries": {
        "metal": Decimal("0.50"),
        "plastic": Decimal("0.20"),
        "electronics": Decimal("0.20"),
        "mixed": Decimal("0.10"),
    },
    "tires": {
        "rubber": Decimal("0.75"),
        "steel": Decimal("0.15"),
        "textile": Decimal("0.05"),
        "mixed": Decimal("0.05"),
    },
    "toys": {
        "plastic": Decimal("0.60"),
        "metal": Decimal("0.15"),
        "paper": Decimal("0.10"),
        "textile": Decimal("0.10"),
        "mixed": Decimal("0.05"),
    },
    "food_products": {
        "organic": Decimal("0.70"),
        "paper": Decimal("0.15"),
        "plastic": Decimal("0.10"),
        "mixed": Decimal("0.05"),
    },
    "beverages": {
        "organic": Decimal("0.50"),
        "glass": Decimal("0.25"),
        "aluminum": Decimal("0.10"),
        "plastic": Decimal("0.10"),
        "mixed": Decimal("0.05"),
    },
    "personal_care": {
        "plastic": Decimal("0.50"),
        "organic": Decimal("0.25"),
        "glass": Decimal("0.10"),
        "paper": Decimal("0.10"),
        "mixed": Decimal("0.05"),
    },
    "cleaning_products": {
        "plastic": Decimal("0.55"),
        "organic": Decimal("0.20"),
        "paper": Decimal("0.15"),
        "mixed": Decimal("0.10"),
    },
    "building_materials": {
        "concrete": Decimal("0.40"),
        "steel": Decimal("0.25"),
        "wood": Decimal("0.20"),
        "glass": Decimal("0.05"),
        "mixed": Decimal("0.10"),
    },
    "paper_products": {
        "paper": Decimal("0.75"),
        "cardboard": Decimal("0.15"),
        "mixed": Decimal("0.10"),
    },
    "glass_containers": {
        "glass": Decimal("0.90"),
        "mixed": Decimal("0.10"),
    },
    "metal_containers": {
        "steel": Decimal("0.50"),
        "aluminum": Decimal("0.40"),
        "mixed": Decimal("0.10"),
    },
    "medical_devices": {
        "plastic": Decimal("0.45"),
        "metal": Decimal("0.30"),
        "glass": Decimal("0.10"),
        "rubber": Decimal("0.05"),
        "mixed": Decimal("0.10"),
    },
    "general_consumer_goods": {
        "plastic": Decimal("0.35"),
        "metal": Decimal("0.20"),
        "paper": Decimal("0.15"),
        "wood": Decimal("0.10"),
        "textile": Decimal("0.10"),
        "mixed": Decimal("0.10"),
    },
}


# ==============================================================================
# REGIONAL TREATMENT MIX PROFILES
# ==============================================================================
# Default end-of-life treatment pathway fractions by region.
# Fractions sum to 1.0 for each region.
# Sources: World Bank What a Waste 2.0, OECD Environment Statistics,
#          Eurostat waste statistics, EPA Facts & Figures, national reports

REGIONAL_TREATMENT_MIXES: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "landfill": Decimal("0.50"),
        "incineration": Decimal("0.14"),
        "recycling": Decimal("0.32"),
        "composting": Decimal("0.04"),
    },
    "EU": {
        "landfill": Decimal("0.24"),
        "incineration": Decimal("0.27"),
        "recycling": Decimal("0.44"),
        "composting": Decimal("0.05"),
    },
    "JP": {
        "landfill": Decimal("0.01"),
        "incineration": Decimal("0.79"),
        "recycling": Decimal("0.19"),
        "composting": Decimal("0.01"),
    },
    "CN": {
        "landfill": Decimal("0.55"),
        "incineration": Decimal("0.30"),
        "recycling": Decimal("0.12"),
        "composting": Decimal("0.03"),
    },
    "IN": {
        "landfill": Decimal("0.75"),
        "incineration": Decimal("0.05"),
        "recycling": Decimal("0.10"),
        "composting": Decimal("0.10"),
    },
    "BR": {
        "landfill": Decimal("0.58"),
        "incineration": Decimal("0.01"),
        "recycling": Decimal("0.14"),
        "composting": Decimal("0.27"),
    },
    "UK": {
        "landfill": Decimal("0.28"),
        "incineration": Decimal("0.28"),
        "recycling": Decimal("0.39"),
        "composting": Decimal("0.05"),
    },
    "AU": {
        "landfill": Decimal("0.40"),
        "incineration": Decimal("0.06"),
        "recycling": Decimal("0.46"),
        "composting": Decimal("0.08"),
    },
    "KR": {
        "landfill": Decimal("0.09"),
        "incineration": Decimal("0.25"),
        "recycling": Decimal("0.60"),
        "composting": Decimal("0.06"),
    },
    "CA": {
        "landfill": Decimal("0.55"),
        "incineration": Decimal("0.04"),
        "recycling": Decimal("0.36"),
        "composting": Decimal("0.05"),
    },
    "MX": {
        "landfill": Decimal("0.70"),
        "incineration": Decimal("0.02"),
        "recycling": Decimal("0.10"),
        "composting": Decimal("0.18"),
    },
    "GLOBAL": {
        "landfill": Decimal("0.40"),
        "incineration": Decimal("0.16"),
        "recycling": Decimal("0.18"),
        "composting": Decimal("0.26"),
    },
}


# ==============================================================================
# IPCC FOD PARAMETERS BY MATERIAL AND CLIMATE ZONE
# ==============================================================================
# Degradable Organic Carbon (DOC), fraction decomposed (DOCf),
# Methane Correction Factor (MCF), decay rate constant (k),
# CH4 fraction in gas (F), oxidation factor (OX).
# Sources: IPCC 2006 Vol 5 Ch 2-3, IPCC 2019 Refinement

# DOC values per material type (fraction of wet mass)
MATERIAL_DOC_VALUES: Dict[str, Decimal] = {
    "plastic": Decimal("0.000"),
    "metal": Decimal("0.000"),
    "aluminum": Decimal("0.000"),
    "steel": Decimal("0.000"),
    "glass": Decimal("0.000"),
    "paper": Decimal("0.400"),
    "cardboard": Decimal("0.400"),
    "wood": Decimal("0.430"),
    "textile": Decimal("0.240"),
    "electronics": Decimal("0.050"),
    "organic": Decimal("0.150"),
    "rubber": Decimal("0.390"),
    "ceramic": Decimal("0.000"),
    "concrete": Decimal("0.000"),
    "mixed": Decimal("0.160"),
}

# DOCf (fraction of DOC that decomposes) - IPCC default
DEFAULT_DOCF: Decimal = Decimal("0.50")

# MCF by landfill type (IPCC 2006 Vol 5 Table 3.1)
MCF_BY_LANDFILL_TYPE: Dict[str, Decimal] = {
    "unmanaged": Decimal("0.40"),
    "managed_anaerobic": Decimal("1.00"),
    "managed_semi_aerobic": Decimal("0.50"),
    "unmanaged_deep": Decimal("0.80"),
    "unmanaged_shallow": Decimal("0.40"),
    "engineered_with_gas": Decimal("1.00"),
}

# Decay rate constants (k, yr^-1) by climate zone and material group
# Only materials with DOC > 0 have meaningful k values.
DECAY_RATE_CONSTANTS: Dict[str, Dict[str, Decimal]] = {
    "boreal_temperate_dry": {
        "paper": Decimal("0.04"),
        "cardboard": Decimal("0.04"),
        "wood": Decimal("0.02"),
        "textile": Decimal("0.04"),
        "organic": Decimal("0.06"),
        "rubber": Decimal("0.04"),
        "electronics": Decimal("0.05"),
        "mixed": Decimal("0.05"),
    },
    "temperate_wet": {
        "paper": Decimal("0.06"),
        "cardboard": Decimal("0.06"),
        "wood": Decimal("0.03"),
        "textile": Decimal("0.06"),
        "organic": Decimal("0.185"),
        "rubber": Decimal("0.06"),
        "electronics": Decimal("0.09"),
        "mixed": Decimal("0.09"),
    },
    "tropical_dry": {
        "paper": Decimal("0.045"),
        "cardboard": Decimal("0.045"),
        "wood": Decimal("0.025"),
        "textile": Decimal("0.045"),
        "organic": Decimal("0.085"),
        "rubber": Decimal("0.045"),
        "electronics": Decimal("0.065"),
        "mixed": Decimal("0.065"),
    },
    "tropical_wet": {
        "paper": Decimal("0.07"),
        "cardboard": Decimal("0.07"),
        "wood": Decimal("0.035"),
        "textile": Decimal("0.07"),
        "organic": Decimal("0.40"),
        "rubber": Decimal("0.07"),
        "electronics": Decimal("0.17"),
        "mixed": Decimal("0.17"),
    },
}

# Default CH4 fraction in landfill gas
DEFAULT_F_CH4: Decimal = Decimal("0.50")

# Default oxidation factors by scenario
DEFAULT_OX_WITH_COVER: Decimal = Decimal("0.10")
DEFAULT_OX_WITHOUT_COVER: Decimal = Decimal("0.00")

# ==============================================================================
# INCINERATION PARAMETERS BY MATERIAL
# ==============================================================================
# Per IPCC 2006 Vol 5 Table 5.2.
# dm = dry matter fraction, cf = carbon fraction of dry matter,
# fcf = fossil carbon fraction, of = oxidation factor (default 1.0)

INCINERATION_PARAMS: Dict[str, Dict[str, Decimal]] = {
    "plastic": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.75"),
        "fossil_carbon_fraction": Decimal("1.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "metal": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "aluminum": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "steel": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "glass": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "paper": {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "oxidation_factor": Decimal("1.00"),
    },
    "cardboard": {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.44"),
        "fossil_carbon_fraction": Decimal("0.01"),
        "oxidation_factor": Decimal("1.00"),
    },
    "wood": {
        "dry_matter_fraction": Decimal("0.85"),
        "carbon_fraction": Decimal("0.50"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "textile": {
        "dry_matter_fraction": Decimal("0.80"),
        "carbon_fraction": Decimal("0.46"),
        "fossil_carbon_fraction": Decimal("0.50"),
        "oxidation_factor": Decimal("1.00"),
    },
    "electronics": {
        "dry_matter_fraction": Decimal("0.90"),
        "carbon_fraction": Decimal("0.20"),
        "fossil_carbon_fraction": Decimal("0.60"),
        "oxidation_factor": Decimal("1.00"),
    },
    "organic": {
        "dry_matter_fraction": Decimal("0.40"),
        "carbon_fraction": Decimal("0.38"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "rubber": {
        "dry_matter_fraction": Decimal("0.84"),
        "carbon_fraction": Decimal("0.67"),
        "fossil_carbon_fraction": Decimal("0.80"),
        "oxidation_factor": Decimal("1.00"),
    },
    "ceramic": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "concrete": {
        "dry_matter_fraction": Decimal("1.00"),
        "carbon_fraction": Decimal("0.00"),
        "fossil_carbon_fraction": Decimal("0.00"),
        "oxidation_factor": Decimal("1.00"),
    },
    "mixed": {
        "dry_matter_fraction": Decimal("0.70"),
        "carbon_fraction": Decimal("0.30"),
        "fossil_carbon_fraction": Decimal("0.40"),
        "oxidation_factor": Decimal("1.00"),
    },
}

# ==============================================================================
# GAS COLLECTION EFFICIENCY BY LANDFILL TYPE
# ==============================================================================
# Fraction of generated CH4 captured by gas collection system.
# Source: IPCC 2006 Vol 5, EPA AP-42

GAS_COLLECTION_EFFICIENCY: Dict[str, Decimal] = {
    "unmanaged": Decimal("0.00"),
    "managed_anaerobic": Decimal("0.20"),
    "managed_semi_aerobic": Decimal("0.10"),
    "unmanaged_deep": Decimal("0.00"),
    "unmanaged_shallow": Decimal("0.00"),
    "engineered_with_gas": Decimal("0.75"),
}

# ==============================================================================
# ENERGY RECOVERY FACTORS BY REGION
# ==============================================================================
# WtE electrical efficiency and displaced grid emission factor.
# wte_efficiency: fraction of thermal energy converted to electricity
# displaced_grid_ef: kgCO2e/kWh of displaced grid electricity

ENERGY_RECOVERY_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.386"),
    },
    "EU": {
        "wte_efficiency": Decimal("0.25"),
        "displaced_grid_ef": Decimal("0.231"),
    },
    "JP": {
        "wte_efficiency": Decimal("0.20"),
        "displaced_grid_ef": Decimal("0.457"),
    },
    "CN": {
        "wte_efficiency": Decimal("0.18"),
        "displaced_grid_ef": Decimal("0.555"),
    },
    "IN": {
        "wte_efficiency": Decimal("0.15"),
        "displaced_grid_ef": Decimal("0.708"),
    },
    "BR": {
        "wte_efficiency": Decimal("0.18"),
        "displaced_grid_ef": Decimal("0.074"),
    },
    "UK": {
        "wte_efficiency": Decimal("0.26"),
        "displaced_grid_ef": Decimal("0.207"),
    },
    "AU": {
        "wte_efficiency": Decimal("0.20"),
        "displaced_grid_ef": Decimal("0.656"),
    },
    "KR": {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.415"),
    },
    "CA": {
        "wte_efficiency": Decimal("0.22"),
        "displaced_grid_ef": Decimal("0.130"),
    },
    "MX": {
        "wte_efficiency": Decimal("0.17"),
        "displaced_grid_ef": Decimal("0.435"),
    },
    "GLOBAL": {
        "wte_efficiency": Decimal("0.20"),
        "displaced_grid_ef": Decimal("0.400"),
    },
}


# ==============================================================================
# RECYCLING PROCESSING EMISSION FACTORS
# ==============================================================================
# kgCO2e/kg: transport to MRF + MRF sorting/processing energy
# These are the ACTUAL emissions under the GHG Protocol cut-off approach.
# Source: DEFRA 2024, EPA WARM v16

RECYCLING_PROCESSING_EFS: Dict[str, Decimal] = {
    "plastic": Decimal("0.035"),
    "metal": Decimal("0.025"),
    "aluminum": Decimal("0.030"),
    "steel": Decimal("0.022"),
    "glass": Decimal("0.018"),
    "paper": Decimal("0.021"),
    "cardboard": Decimal("0.020"),
    "wood": Decimal("0.015"),
    "textile": Decimal("0.028"),
    "electronics": Decimal("0.045"),
    "organic": Decimal("0.010"),
    "rubber": Decimal("0.032"),
    "ceramic": Decimal("0.012"),
    "concrete": Decimal("0.008"),
    "mixed": Decimal("0.025"),
}


# ==============================================================================
# AVOIDED EMISSION FACTORS (VIRGIN MATERIAL SUBSTITUTION)
# ==============================================================================
# kgCO2e/kg avoided when recycled material displaces virgin production.
# ALWAYS reported SEPARATELY, NEVER netted from total per GHG Protocol.
# Negative values indicate emission REDUCTIONS (credit).
# Source: EPA WARM v16, DEFRA 2024, Ecoinvent

AVOIDED_EMISSION_FACTORS: Dict[str, Decimal] = {
    "plastic": Decimal("-1.440"),
    "metal": Decimal("-4.200"),
    "aluminum": Decimal("-9.120"),
    "steel": Decimal("-1.820"),
    "glass": Decimal("-0.314"),
    "paper": Decimal("-0.840"),
    "cardboard": Decimal("-0.720"),
    "wood": Decimal("-0.580"),
    "textile": Decimal("-3.100"),
    "electronics": Decimal("-2.950"),
    "organic": Decimal("0.000"),
    "rubber": Decimal("-1.220"),
    "ceramic": Decimal("-0.150"),
    "concrete": Decimal("-0.024"),
    "mixed": Decimal("-0.680"),
}


# ==============================================================================
# PRODUCT WEIGHT DEFAULTS
# ==============================================================================
# Average product weight in kg per unit.
# Used when unit weight is not provided by the reporting company.
# Source: Industry averages, product characterization studies, EPA reports

PRODUCT_WEIGHT_DEFAULTS: Dict[str, Decimal] = {
    "electronics": Decimal("2.50"),
    "appliances": Decimal("35.00"),
    "packaging": Decimal("0.15"),
    "clothing": Decimal("0.50"),
    "footwear": Decimal("0.80"),
    "furniture": Decimal("25.00"),
    "automotive_parts": Decimal("5.00"),
    "batteries": Decimal("0.25"),
    "tires": Decimal("9.00"),
    "toys": Decimal("0.40"),
    "food_products": Decimal("0.50"),
    "beverages": Decimal("0.35"),
    "personal_care": Decimal("0.20"),
    "cleaning_products": Decimal("1.00"),
    "building_materials": Decimal("50.00"),
    "paper_products": Decimal("0.10"),
    "glass_containers": Decimal("0.30"),
    "metal_containers": Decimal("0.05"),
    "medical_devices": Decimal("0.50"),
    "general_consumer_goods": Decimal("1.00"),
}


# ==============================================================================
# COMPOSTING EMISSION FACTORS
# ==============================================================================
# Per IPCC 2006 Vol 5 Ch 4.
# CH4 and N2O emission factors in kg per kg of waste.

COMPOSTING_EFS: Dict[str, Decimal] = {
    "ch4_ef_kg_per_kg": Decimal("0.004"),     # 4 g CH4/kg
    "n2o_ef_kg_per_kg": Decimal("0.0003"),    # 0.3 g N2O/kg
    "ch4_ef_dry_kg_per_kg": Decimal("0.010"), # 10 g CH4/kg dry
    "n2o_ef_dry_kg_per_kg": Decimal("0.0006"), # 0.6 g N2O/kg dry
}


# ==============================================================================
# ANAEROBIC DIGESTION EMISSION FACTORS
# ==============================================================================
# Per IPCC 2019 Refinement.
# Biogas yield, CH4 content, capture efficiency

ANAEROBIC_DIGESTION_EFS: Dict[str, Any] = {
    "biogas_yield_m3_per_kg": Decimal("0.10"),   # Average m3 biogas/kg waste
    "ch4_fraction_in_biogas": Decimal("0.60"),     # 60% CH4 in biogas
    "default_capture_efficiency": Decimal("0.98"), # 98% capture (enclosed digester)
    "leakage_rate_enclosed": Decimal("0.02"),      # 2% leakage
    "leakage_rate_open": Decimal("0.07"),          # 7% leakage (open digestate)
}


# ==============================================================================
# OPEN BURNING EMISSION FACTORS
# ==============================================================================
# kgCO2e/kg: CO2 + CH4 + N2O combined factor for uncontrolled burning.
# Source: IPCC 2006 Vol 5 Ch 5 Table 5.5

OPEN_BURNING_EFS: Dict[str, Dict[str, Decimal]] = {
    "plastic": {
        "co2_fossil_kg_per_kg": Decimal("2.900"),
        "ch4_kg_per_kg": Decimal("0.0060"),
        "n2o_kg_per_kg": Decimal("0.00015"),
    },
    "paper": {
        "co2_fossil_kg_per_kg": Decimal("0.000"),
        "co2_biogenic_kg_per_kg": Decimal("1.410"),
        "ch4_kg_per_kg": Decimal("0.0027"),
        "n2o_kg_per_kg": Decimal("0.00010"),
    },
    "cardboard": {
        "co2_fossil_kg_per_kg": Decimal("0.000"),
        "co2_biogenic_kg_per_kg": Decimal("1.170"),
        "ch4_kg_per_kg": Decimal("0.0027"),
        "n2o_kg_per_kg": Decimal("0.00010"),
    },
    "wood": {
        "co2_fossil_kg_per_kg": Decimal("0.000"),
        "co2_biogenic_kg_per_kg": Decimal("1.500"),
        "ch4_kg_per_kg": Decimal("0.0027"),
        "n2o_kg_per_kg": Decimal("0.00010"),
    },
    "textile": {
        "co2_fossil_kg_per_kg": Decimal("1.350"),
        "co2_biogenic_kg_per_kg": Decimal("1.100"),
        "ch4_kg_per_kg": Decimal("0.0040"),
        "n2o_kg_per_kg": Decimal("0.00012"),
    },
    "organic": {
        "co2_fossil_kg_per_kg": Decimal("0.000"),
        "co2_biogenic_kg_per_kg": Decimal("0.550"),
        "ch4_kg_per_kg": Decimal("0.0040"),
        "n2o_kg_per_kg": Decimal("0.00015"),
    },
    "rubber": {
        "co2_fossil_kg_per_kg": Decimal("2.800"),
        "ch4_kg_per_kg": Decimal("0.0050"),
        "n2o_kg_per_kg": Decimal("0.00015"),
    },
    "mixed": {
        "co2_fossil_kg_per_kg": Decimal("0.900"),
        "co2_biogenic_kg_per_kg": Decimal("0.500"),
        "ch4_kg_per_kg": Decimal("0.0035"),
        "n2o_kg_per_kg": Decimal("0.00012"),
    },
}


# ==============================================================================
# GWP VALUES (IPCC Assessment Report versions)
# ==============================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    "AR5": {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    "AR6": {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
    "AR6_20yr": {"co2": Decimal("1"), "ch4": Decimal("82.5"), "n2o": Decimal("273")},
}


# ==============================================================================
# N2O EMISSION FACTOR FOR MSW INCINERATION
# ==============================================================================
# kg N2O per kg of waste incinerated (MSW default)
# Source: IPCC 2006 Vol 5 Table 5.3

DEFAULT_N2O_INCINERATION_EF: Decimal = Decimal("0.00004")  # 40 g/tonne = 0.04 kg/t


# ==============================================================================
# CALORIFIC VALUES BY MATERIAL
# ==============================================================================
# Net calorific value in MJ/kg for energy recovery calculations.
# Source: IPCC 2006 Vol 5 Table 5.6

CALORIFIC_VALUES: Dict[str, Decimal] = {
    "plastic": Decimal("32.0"),
    "metal": Decimal("0.0"),
    "aluminum": Decimal("0.0"),
    "steel": Decimal("0.0"),
    "glass": Decimal("0.0"),
    "paper": Decimal("12.5"),
    "cardboard": Decimal("11.8"),
    "wood": Decimal("15.0"),
    "textile": Decimal("16.0"),
    "electronics": Decimal("8.0"),
    "organic": Decimal("4.0"),
    "rubber": Decimal("23.0"),
    "ceramic": Decimal("0.0"),
    "concrete": Decimal("0.0"),
    "mixed": Decimal("10.0"),
}


# ==============================================================================
# EOL PRODUCT DATABASE ENGINE - SINGLETON
# ==============================================================================


class EOLProductDatabaseEngine:
    """
    Thread-safe singleton engine for end-of-life treatment emission factor lookups.

    This engine provides zero-hallucination deterministic lookups for:
    - Material-level emission factors by treatment pathway
    - Product category material compositions (bill of materials)
    - Regional treatment mix profiles (12 regions)
    - IPCC FOD landfill parameters (DOC, DOCf, MCF, k, F, OX)
    - Incineration parameters (dm, CF, FCF, OF)
    - Gas collection efficiency by landfill type
    - Energy recovery factors (WtE efficiency, displaced grid EF)
    - Recycling processing emission factors
    - Avoided emission factors (virgin substitution credits)
    - Product weight defaults (kg per unit)
    - Composting and anaerobic digestion parameters

    All values are returned as Decimal for regulatory-grade precision.
    No LLM calls are made anywhere in this engine.

    Thread Safety:
        Uses __new__ singleton with threading.Lock() and threading.RLock()
        for all mutable state access.

    Performance:
        All data is embedded in-memory dictionaries for O(1) lookups.
        time.monotonic() used for operation timing.

    Provenance:
        SHA-256 hashing of lookup inputs and outputs for audit trail.

    Attributes:
        _lookup_count: Total number of factor lookups performed.
        _last_access: Timestamp of last access.

    Example:
        >>> engine = EOLProductDatabaseEngine()
        >>> ef = engine.get_material_ef("plastic", "incineration")
        >>> ef
        Decimal('2.760')
        >>> comp = engine.get_product_composition("electronics")
        >>> weight = engine.get_product_weight("electronics")
    """

    _instance: Optional["EOLProductDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EOLProductDatabaseEngine":
        """Thread-safe singleton instantiation via double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the EOLProductDatabaseEngine (called once due to singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        start_ts = time.monotonic()
        self._initialized: bool = True
        self._rlock: threading.RLock = threading.RLock()
        self._lookup_count: int = 0
        self._last_access: Optional[datetime] = None

        # Validate embedded data integrity on initialization
        self._validate_data_integrity()

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.info(
            "EOLProductDatabaseEngine initialized: "
            "materials=%d, products=%d, regions=%d, "
            "climate_zones=%d, landfill_types=%d, elapsed_ms=%.2f",
            len(MATERIAL_EMISSION_FACTORS),
            len(PRODUCT_COMPOSITIONS),
            len(REGIONAL_TREATMENT_MIXES),
            len(DECAY_RATE_CONSTANTS),
            len(GAS_COLLECTION_EFFICIENCY),
            elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        This method is intended exclusively for unit tests that need
        a fresh engine instance. It should never be called in production.
        """
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Data integrity validation
    # ------------------------------------------------------------------

    def _validate_data_integrity(self) -> None:
        """
        Validate all embedded reference data for internal consistency.

        Checks:
        - All product compositions sum to 1.0 (within tolerance)
        - All regional treatment mixes sum to 1.0 (within tolerance)
        - All composition materials exist in MATERIAL_EMISSION_FACTORS
        - All regional treatment types exist in VALID_TREATMENTS
        - All materials have incineration parameters
        - All materials have recycling processing EFs

        Raises:
            ValueError: If any data integrity check fails.
        """
        tolerance = Decimal("0.001")

        # Validate product compositions sum to 1.0
        for category, composition in PRODUCT_COMPOSITIONS.items():
            total = sum(composition.values())
            if abs(total - _ONE) > tolerance:
                raise ValueError(
                    f"Product composition for '{category}' sums to {total}, expected 1.0"
                )
            for material in composition:
                if material not in VALID_MATERIALS:
                    raise ValueError(
                        f"Unknown material '{material}' in product '{category}'"
                    )

        # Validate regional treatment mixes sum to 1.0
        for region, mix in REGIONAL_TREATMENT_MIXES.items():
            total = sum(mix.values())
            if abs(total - _ONE) > tolerance:
                raise ValueError(
                    f"Regional mix for '{region}' sums to {total}, expected 1.0"
                )

        # Validate all materials have required databases
        for material in VALID_MATERIALS:
            if material not in MATERIAL_EMISSION_FACTORS:
                raise ValueError(
                    f"Material '{material}' missing from MATERIAL_EMISSION_FACTORS"
                )
            if material not in INCINERATION_PARAMS:
                raise ValueError(
                    f"Material '{material}' missing from INCINERATION_PARAMS"
                )
            if material not in RECYCLING_PROCESSING_EFS:
                raise ValueError(
                    f"Material '{material}' missing from RECYCLING_PROCESSING_EFS"
                )

        logger.debug("Data integrity validation passed for all embedded databases.")

    def _increment_lookup_count(self) -> None:
        """Thread-safe increment of lookup counter."""
        with self._rlock:
            self._lookup_count += 1
            self._last_access = datetime.now(timezone.utc)

    # ==================================================================
    # 1. MATERIAL EMISSION FACTOR LOOKUPS
    # ==================================================================

    def get_material_ef(
        self,
        material: str,
        treatment: str,
    ) -> Decimal:
        """
        Get emission factor for a specific material and treatment pathway.

        Args:
            material: Material type (e.g., "plastic", "paper", "aluminum").
                     Must be one of VALID_MATERIALS.
            treatment: Treatment pathway (e.g., "landfill", "incineration").
                      Must be one of VALID_TREATMENTS.

        Returns:
            Emission factor in kgCO2e/kg. Negative values for recycling
            indicate avoided emissions (virgin substitution credit).

        Raises:
            ValueError: If material or treatment is not valid.

        Example:
            >>> engine = EOLProductDatabaseEngine()
            >>> engine.get_material_ef("plastic", "landfill")
            Decimal('0.021')
            >>> engine.get_material_ef("aluminum", "recycling")
            Decimal('-9.120')
        """
        start_ts = time.monotonic()
        self._increment_lookup_count()

        material_lower = material.lower().strip()
        treatment_lower = treatment.lower().strip()

        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )
        if treatment_lower not in VALID_TREATMENTS:
            raise ValueError(
                f"Invalid treatment '{treatment}'. Must be one of: {sorted(VALID_TREATMENTS)}"
            )

        ef = MATERIAL_EMISSION_FACTORS[material_lower][treatment_lower]

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "get_material_ef: material=%s, treatment=%s, ef=%s kgCO2e/kg, elapsed_ms=%.3f",
            material_lower, treatment_lower, ef, elapsed_ms,
        )
        return ef

    def get_all_material_efs(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get the complete material emission factor database.

        Returns:
            Deep copy of the full emission factor dictionary,
            keyed by material then by treatment pathway.

        Example:
            >>> efs = engine.get_all_material_efs()
            >>> efs["plastic"]["incineration"]
            Decimal('2.760')
        """
        self._increment_lookup_count()
        # Return deep copy to prevent mutation
        return {
            material: dict(treatments)
            for material, treatments in MATERIAL_EMISSION_FACTORS.items()
        }

    def get_material_efs_for_treatment(
        self,
        treatment: str,
    ) -> Dict[str, Decimal]:
        """
        Get all material EFs for a specific treatment pathway.

        Args:
            treatment: Treatment pathway (e.g., "landfill").

        Returns:
            Dict mapping material name to EF (kgCO2e/kg).

        Raises:
            ValueError: If treatment is not valid.

        Example:
            >>> efs = engine.get_material_efs_for_treatment("landfill")
            >>> efs["plastic"]
            Decimal('0.021')
        """
        self._increment_lookup_count()
        treatment_lower = treatment.lower().strip()

        if treatment_lower not in VALID_TREATMENTS:
            raise ValueError(
                f"Invalid treatment '{treatment}'. Must be one of: {sorted(VALID_TREATMENTS)}"
            )

        return {
            material: treatments[treatment_lower]
            for material, treatments in MATERIAL_EMISSION_FACTORS.items()
        }

    # ==================================================================
    # 2. PRODUCT COMPOSITION LOOKUPS
    # ==================================================================

    def get_product_composition(
        self,
        category: str,
    ) -> Dict[str, Decimal]:
        """
        Get default material composition (BOM) for a product category.

        Args:
            category: Product category (e.g., "electronics", "packaging").
                     Must be one of VALID_PRODUCT_CATEGORIES.

        Returns:
            Dict mapping material name to mass fraction.
            Fractions sum to 1.0.

        Raises:
            ValueError: If category is not valid.

        Example:
            >>> comp = engine.get_product_composition("electronics")
            >>> comp
            {'plastic': Decimal('0.40'), 'metal': Decimal('0.35'), ...}
            >>> sum(comp.values())
            Decimal('1.00')
        """
        start_ts = time.monotonic()
        self._increment_lookup_count()

        category_lower = category.lower().strip()
        if category_lower not in VALID_PRODUCT_CATEGORIES:
            raise ValueError(
                f"Invalid product category '{category}'. Must be one of: "
                f"{sorted(VALID_PRODUCT_CATEGORIES)}"
            )

        composition = dict(PRODUCT_COMPOSITIONS[category_lower])

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "get_product_composition: category=%s, materials=%d, elapsed_ms=%.3f",
            category_lower, len(composition), elapsed_ms,
        )
        return composition

    def get_all_compositions(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all product category compositions.

        Returns:
            Deep copy of the full product composition dictionary.

        Example:
            >>> all_comp = engine.get_all_compositions()
            >>> len(all_comp)
            20
        """
        self._increment_lookup_count()
        return {
            category: dict(composition)
            for category, composition in PRODUCT_COMPOSITIONS.items()
        }

    # ==================================================================
    # 3. REGIONAL TREATMENT MIX LOOKUPS
    # ==================================================================

    def get_regional_treatment_mix(
        self,
        region: str,
    ) -> Dict[str, Decimal]:
        """
        Get default end-of-life treatment pathway mix for a region.

        Args:
            region: Region code (e.g., "US", "EU", "JP", "GLOBAL").
                   Must be one of VALID_REGIONS.

        Returns:
            Dict mapping treatment pathway to fraction.
            Fractions sum to 1.0.

        Raises:
            ValueError: If region is not valid.

        Example:
            >>> mix = engine.get_regional_treatment_mix("US")
            >>> mix
            {'landfill': Decimal('0.50'), 'incineration': Decimal('0.14'), ...}
        """
        start_ts = time.monotonic()
        self._increment_lookup_count()

        region_upper = region.upper().strip()
        if region_upper not in VALID_REGIONS:
            raise ValueError(
                f"Invalid region '{region}'. Must be one of: {sorted(VALID_REGIONS)}"
            )

        mix = dict(REGIONAL_TREATMENT_MIXES[region_upper])

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "get_regional_treatment_mix: region=%s, pathways=%d, elapsed_ms=%.3f",
            region_upper, len(mix), elapsed_ms,
        )
        return mix

    def get_all_regional_mixes(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all regional treatment mix profiles.

        Returns:
            Deep copy of the full regional treatment mix dictionary.

        Example:
            >>> all_mixes = engine.get_all_regional_mixes()
            >>> len(all_mixes)
            12
        """
        self._increment_lookup_count()
        return {
            region: dict(mix)
            for region, mix in REGIONAL_TREATMENT_MIXES.items()
        }

    # ==================================================================
    # 4. LANDFILL FOD PARAMETER LOOKUPS
    # ==================================================================

    def get_landfill_fod_params(
        self,
        material: str,
        climate_zone: str,
    ) -> Dict[str, Decimal]:
        """
        Get IPCC First Order Decay (FOD) parameters for landfill modelling.

        Returns DOC, DOCf, MCF (for managed_anaerobic default), decay rate k,
        CH4 fraction F, and default oxidation factor OX.

        Args:
            material: Material type (e.g., "paper", "organic").
            climate_zone: IPCC climate zone (e.g., "temperate_wet").

        Returns:
            Dict with keys: doc, docf, k, f_ch4, ox_with_cover, ox_without_cover.
            Materials with DOC=0 (metals, glass, etc.) return zero for all
            decomposition parameters.

        Raises:
            ValueError: If material or climate_zone is not valid.

        Example:
            >>> params = engine.get_landfill_fod_params("paper", "temperate_wet")
            >>> params["doc"]
            Decimal('0.400')
            >>> params["k"]
            Decimal('0.06')
        """
        start_ts = time.monotonic()
        self._increment_lookup_count()

        material_lower = material.lower().strip()
        climate_lower = climate_zone.lower().strip()

        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )
        if climate_lower not in VALID_CLIMATE_ZONES:
            raise ValueError(
                f"Invalid climate zone '{climate_zone}'. Must be one of: "
                f"{sorted(VALID_CLIMATE_ZONES)}"
            )

        doc = MATERIAL_DOC_VALUES.get(material_lower, _ZERO)

        # Get decay rate k: only meaningful for materials with DOC > 0
        k_value = _ZERO
        if doc > _ZERO and climate_lower in DECAY_RATE_CONSTANTS:
            k_value = DECAY_RATE_CONSTANTS[climate_lower].get(material_lower, _ZERO)

        result = {
            "doc": doc,
            "docf": DEFAULT_DOCF,
            "k": k_value,
            "f_ch4": DEFAULT_F_CH4,
            "ox_with_cover": DEFAULT_OX_WITH_COVER,
            "ox_without_cover": DEFAULT_OX_WITHOUT_COVER,
        }

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "get_landfill_fod_params: material=%s, climate=%s, doc=%s, k=%s, elapsed_ms=%.3f",
            material_lower, climate_lower, doc, k_value, elapsed_ms,
        )
        return result

    def get_mcf(self, landfill_type: str) -> Decimal:
        """
        Get Methane Correction Factor (MCF) for a landfill type.

        Args:
            landfill_type: Landfill management type.

        Returns:
            MCF value (Decimal, 0.0-1.0).

        Raises:
            ValueError: If landfill_type is not valid.

        Example:
            >>> engine.get_mcf("managed_anaerobic")
            Decimal('1.00')
        """
        self._increment_lookup_count()
        lt_lower = landfill_type.lower().strip()

        if lt_lower not in VALID_LANDFILL_TYPES:
            raise ValueError(
                f"Invalid landfill type '{landfill_type}'. "
                f"Must be one of: {sorted(VALID_LANDFILL_TYPES)}"
            )

        return MCF_BY_LANDFILL_TYPE[lt_lower]

    def get_doc(self, material: str) -> Decimal:
        """
        Get Degradable Organic Carbon (DOC) fraction for a material.

        Args:
            material: Material type.

        Returns:
            DOC value (Decimal, 0.0-0.43).

        Raises:
            ValueError: If material is not valid.
        """
        self._increment_lookup_count()
        material_lower = material.lower().strip()
        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )
        return MATERIAL_DOC_VALUES[material_lower]

    # ==================================================================
    # 5. INCINERATION PARAMETER LOOKUPS
    # ==================================================================

    def get_incineration_params(
        self,
        material: str,
    ) -> Dict[str, Decimal]:
        """
        Get IPCC incineration parameters for a material type.

        Args:
            material: Material type (e.g., "plastic", "paper").

        Returns:
            Dict with keys: dry_matter_fraction, carbon_fraction,
            fossil_carbon_fraction, oxidation_factor.

        Raises:
            ValueError: If material is not valid.

        Example:
            >>> params = engine.get_incineration_params("plastic")
            >>> params["fossil_carbon_fraction"]
            Decimal('1.00')
        """
        start_ts = time.monotonic()
        self._increment_lookup_count()

        material_lower = material.lower().strip()
        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )

        params = dict(INCINERATION_PARAMS[material_lower])

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "get_incineration_params: material=%s, dm=%s, cf=%s, fcf=%s, elapsed_ms=%.3f",
            material_lower,
            params["dry_matter_fraction"],
            params["carbon_fraction"],
            params["fossil_carbon_fraction"],
            elapsed_ms,
        )
        return params

    def get_calorific_value(self, material: str) -> Decimal:
        """
        Get net calorific value for a material (MJ/kg).

        Used for energy recovery calculations in WtE plants.

        Args:
            material: Material type.

        Returns:
            Net calorific value in MJ/kg.

        Raises:
            ValueError: If material is not valid.
        """
        self._increment_lookup_count()
        material_lower = material.lower().strip()
        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )
        return CALORIFIC_VALUES.get(material_lower, _ZERO)

    def get_n2o_incineration_ef(self) -> Decimal:
        """
        Get default N2O emission factor for MSW incineration.

        Returns:
            N2O EF in kg N2O per kg of waste incinerated.
        """
        self._increment_lookup_count()
        return DEFAULT_N2O_INCINERATION_EF

    # ==================================================================
    # 6. GAS COLLECTION EFFICIENCY LOOKUPS
    # ==================================================================

    def get_gas_collection_efficiency(
        self,
        landfill_type: str,
    ) -> Decimal:
        """
        Get landfill gas collection efficiency for a landfill type.

        Args:
            landfill_type: Landfill management type (e.g., "engineered_with_gas").

        Returns:
            Gas collection efficiency fraction (0.0-0.75).

        Raises:
            ValueError: If landfill_type is not valid.

        Example:
            >>> engine.get_gas_collection_efficiency("engineered_with_gas")
            Decimal('0.75')
            >>> engine.get_gas_collection_efficiency("unmanaged")
            Decimal('0.00')
        """
        self._increment_lookup_count()
        lt_lower = landfill_type.lower().strip()

        if lt_lower not in VALID_LANDFILL_TYPES:
            raise ValueError(
                f"Invalid landfill type '{landfill_type}'. "
                f"Must be one of: {sorted(VALID_LANDFILL_TYPES)}"
            )

        return GAS_COLLECTION_EFFICIENCY[lt_lower]

    # ==================================================================
    # 7. ENERGY RECOVERY FACTOR LOOKUPS
    # ==================================================================

    def get_energy_recovery_factor(
        self,
        region: str,
    ) -> Dict[str, Decimal]:
        """
        Get waste-to-energy (WtE) recovery factors for a region.

        Args:
            region: Region code (e.g., "US", "EU", "GLOBAL").

        Returns:
            Dict with keys: wte_efficiency, displaced_grid_ef.

        Raises:
            ValueError: If region is not valid.

        Example:
            >>> factors = engine.get_energy_recovery_factor("US")
            >>> factors["wte_efficiency"]
            Decimal('0.22')
            >>> factors["displaced_grid_ef"]
            Decimal('0.386')
        """
        self._increment_lookup_count()
        region_upper = region.upper().strip()

        if region_upper not in VALID_REGIONS:
            raise ValueError(
                f"Invalid region '{region}'. Must be one of: {sorted(VALID_REGIONS)}"
            )

        return dict(ENERGY_RECOVERY_FACTORS[region_upper])

    # ==================================================================
    # 8. RECYCLING PROCESSING EF LOOKUPS
    # ==================================================================

    def get_recycling_processing_ef(
        self,
        material: str,
    ) -> Decimal:
        """
        Get recycling processing emission factor (transport + MRF energy).

        These are the ACTUAL emissions under the GHG Protocol cut-off approach.
        They represent transport to the recycling facility and MRF sorting energy.
        Avoided emissions from displacing virgin material are reported separately.

        Args:
            material: Material type.

        Returns:
            Processing EF in kgCO2e/kg.

        Raises:
            ValueError: If material is not valid.

        Example:
            >>> engine.get_recycling_processing_ef("plastic")
            Decimal('0.035')
        """
        self._increment_lookup_count()
        material_lower = material.lower().strip()

        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )

        return RECYCLING_PROCESSING_EFS[material_lower]

    # ==================================================================
    # 9. AVOIDED EMISSION FACTOR LOOKUPS
    # ==================================================================

    def get_avoided_emission_factor(
        self,
        material: str,
    ) -> Decimal:
        """
        Get avoided emission factor for virgin material substitution.

        These are credits from displacing virgin material production when
        materials are recycled. They are ALWAYS reported separately per
        the GHG Protocol, NEVER netted from total reported emissions.

        Args:
            material: Material type.

        Returns:
            Avoided EF in kgCO2e/kg (negative values = emission reduction).

        Raises:
            ValueError: If material is not valid.

        Example:
            >>> engine.get_avoided_emission_factor("aluminum")
            Decimal('-9.120')
        """
        self._increment_lookup_count()
        material_lower = material.lower().strip()

        if material_lower not in VALID_MATERIALS:
            raise ValueError(
                f"Invalid material '{material}'. Must be one of: {sorted(VALID_MATERIALS)}"
            )

        return AVOIDED_EMISSION_FACTORS[material_lower]

    def get_all_avoided_factors(self) -> Dict[str, Decimal]:
        """
        Get all avoided emission factors.

        Returns:
            Dict mapping material to avoided EF (kgCO2e/kg).
        """
        self._increment_lookup_count()
        return dict(AVOIDED_EMISSION_FACTORS)

    # ==================================================================
    # 10. PRODUCT WEIGHT LOOKUPS
    # ==================================================================

    def get_product_weight(
        self,
        category: str,
    ) -> Decimal:
        """
        Get default product weight in kg per unit.

        Used when the reporting company does not provide specific
        product weights. Average industry values based on EPA and
        industry characterization studies.

        Args:
            category: Product category.

        Returns:
            Average weight in kg per unit.

        Raises:
            ValueError: If category is not valid.

        Example:
            >>> engine.get_product_weight("electronics")
            Decimal('2.50')
            >>> engine.get_product_weight("packaging")
            Decimal('0.15')
        """
        self._increment_lookup_count()
        category_lower = category.lower().strip()

        if category_lower not in VALID_PRODUCT_CATEGORIES:
            raise ValueError(
                f"Invalid product category '{category}'. Must be one of: "
                f"{sorted(VALID_PRODUCT_CATEGORIES)}"
            )

        return PRODUCT_WEIGHT_DEFAULTS[category_lower]

    def get_all_product_weights(self) -> Dict[str, Decimal]:
        """
        Get all default product weights.

        Returns:
            Dict mapping product category to weight in kg.
        """
        self._increment_lookup_count()
        return dict(PRODUCT_WEIGHT_DEFAULTS)

    # ==================================================================
    # 11. COMPOSTING AND AD LOOKUPS
    # ==================================================================

    def get_composting_ef(self) -> Dict[str, Decimal]:
        """
        Get composting emission factors (CH4 and N2O per kg of waste).

        Returns:
            Dict with keys: ch4_ef_kg_per_kg, n2o_ef_kg_per_kg,
            ch4_ef_dry_kg_per_kg, n2o_ef_dry_kg_per_kg.

        Example:
            >>> efs = engine.get_composting_ef()
            >>> efs["ch4_ef_kg_per_kg"]
            Decimal('0.004')
        """
        self._increment_lookup_count()
        return dict(COMPOSTING_EFS)

    def get_ad_ef(self) -> Dict[str, Any]:
        """
        Get anaerobic digestion emission factors and parameters.

        Returns:
            Dict with biogas_yield_m3_per_kg, ch4_fraction_in_biogas,
            default_capture_efficiency, leakage_rate_enclosed,
            leakage_rate_open.

        Example:
            >>> efs = engine.get_ad_ef()
            >>> efs["ch4_fraction_in_biogas"]
            Decimal('0.60')
        """
        self._increment_lookup_count()
        return dict(ANAEROBIC_DIGESTION_EFS)

    # ==================================================================
    # 12. OPEN BURNING LOOKUPS
    # ==================================================================

    def get_open_burning_ef(
        self,
        material: str,
    ) -> Dict[str, Decimal]:
        """
        Get open burning emission factors by gas type for a material.

        Args:
            material: Material type.

        Returns:
            Dict with co2_fossil_kg_per_kg, co2_biogenic_kg_per_kg (if applicable),
            ch4_kg_per_kg, n2o_kg_per_kg.

        Raises:
            ValueError: If material not found in open burning database.
        """
        self._increment_lookup_count()
        material_lower = material.lower().strip()

        if material_lower not in OPEN_BURNING_EFS:
            # Fall back to mixed waste factors
            logger.warning(
                "No open burning EF for material '%s', using 'mixed' default.",
                material,
            )
            return dict(OPEN_BURNING_EFS["mixed"])

        return dict(OPEN_BURNING_EFS[material_lower])

    # ==================================================================
    # 13. GWP LOOKUPS
    # ==================================================================

    def get_gwp(
        self,
        gas: str,
        ar_version: str = "AR5",
    ) -> Decimal:
        """
        Get Global Warming Potential for a greenhouse gas.

        Args:
            gas: Gas name ("co2", "ch4", "n2o").
            ar_version: IPCC Assessment Report version ("AR4", "AR5", "AR6", "AR6_20yr").

        Returns:
            GWP value (Decimal).

        Raises:
            ValueError: If gas or AR version is not valid.

        Example:
            >>> engine.get_gwp("ch4", "AR5")
            Decimal('28')
        """
        self._increment_lookup_count()
        gas_lower = gas.lower().strip()
        ar_upper = ar_version.upper().strip()

        if ar_upper not in GWP_VALUES:
            raise ValueError(
                f"Invalid AR version '{ar_version}'. "
                f"Must be one of: {sorted(GWP_VALUES.keys())}"
            )

        gwp_table = GWP_VALUES[ar_upper]
        if gas_lower not in gwp_table:
            raise ValueError(
                f"Invalid gas '{gas}'. Must be one of: {sorted(gwp_table.keys())}"
            )

        return gwp_table[gas_lower]

    def get_gwp_ch4(self, ar_version: str = "AR5") -> Decimal:
        """Shorthand to get CH4 GWP for a given AR version."""
        return self.get_gwp("ch4", ar_version)

    def get_gwp_n2o(self, ar_version: str = "AR5") -> Decimal:
        """Shorthand to get N2O GWP for a given AR version."""
        return self.get_gwp("n2o", ar_version)

    # ==================================================================
    # 14. COMPOSITE / CONVENIENCE LOOKUPS
    # ==================================================================

    def lookup_composite(
        self,
        product_category: str,
        region: str,
    ) -> Dict[str, Any]:
        """
        Combined lookup: product composition + regional treatment mix + EFs.

        Provides all data needed for a waste-type-specific calculation for
        a product category in a given region, including per-material EFs
        for each treatment in the regional mix.

        Args:
            product_category: Product category (e.g., "electronics").
            region: Region code (e.g., "US").

        Returns:
            Dict with keys: product_category, region, composition,
            treatment_mix, material_treatment_efs, product_weight_kg.

        Raises:
            ValueError: If product_category or region is not valid.

        Example:
            >>> composite = engine.lookup_composite("electronics", "US")
            >>> composite["composition"]["plastic"]
            Decimal('0.40')
            >>> composite["treatment_mix"]["landfill"]
            Decimal('0.50')
        """
        start_ts = time.monotonic()

        composition = self.get_product_composition(product_category)
        treatment_mix = self.get_regional_treatment_mix(region)
        weight = self.get_product_weight(product_category)

        # Build material x treatment EF matrix
        material_treatment_efs: Dict[str, Dict[str, Decimal]] = {}
        for material in composition:
            material_treatment_efs[material] = {}
            for treatment in treatment_mix:
                if treatment in VALID_TREATMENTS:
                    ef = MATERIAL_EMISSION_FACTORS.get(material, {}).get(treatment, _ZERO)
                    material_treatment_efs[material][treatment] = ef

        result = {
            "product_category": product_category.lower().strip(),
            "region": region.upper().strip(),
            "composition": composition,
            "treatment_mix": treatment_mix,
            "material_treatment_efs": material_treatment_efs,
            "product_weight_kg": weight,
        }

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        logger.debug(
            "lookup_composite: category=%s, region=%s, materials=%d, treatments=%d, "
            "elapsed_ms=%.3f",
            product_category, region, len(composition), len(treatment_mix), elapsed_ms,
        )
        return result

    def lookup_material_treatment_matrix(
        self,
        materials: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Get a subset of the material x treatment EF matrix.

        Args:
            materials: List of materials. If None, returns all.
            treatments: List of treatments. If None, returns all.

        Returns:
            Nested dict: material -> treatment -> EF (kgCO2e/kg).
        """
        self._increment_lookup_count()
        mat_list = materials or list(VALID_MATERIALS)
        trt_list = treatments or list(VALID_TREATMENTS)

        result: Dict[str, Dict[str, Decimal]] = {}
        for mat in mat_list:
            mat_lower = mat.lower().strip()
            if mat_lower not in MATERIAL_EMISSION_FACTORS:
                continue
            result[mat_lower] = {}
            for trt in trt_list:
                trt_lower = trt.lower().strip()
                if trt_lower in MATERIAL_EMISSION_FACTORS[mat_lower]:
                    result[mat_lower][trt_lower] = MATERIAL_EMISSION_FACTORS[mat_lower][trt_lower]

        return result

    # ==================================================================
    # 15. VALIDATION METHODS
    # ==================================================================

    def validate_material(self, material: str) -> bool:
        """
        Check if a material type is valid.

        Args:
            material: Material type to validate.

        Returns:
            True if valid, False otherwise.
        """
        return material.lower().strip() in VALID_MATERIALS

    def validate_treatment(self, treatment: str) -> bool:
        """
        Check if a treatment pathway is valid.

        Args:
            treatment: Treatment pathway to validate.

        Returns:
            True if valid, False otherwise.
        """
        return treatment.lower().strip() in VALID_TREATMENTS

    def validate_product_category(self, category: str) -> bool:
        """
        Check if a product category is valid.

        Args:
            category: Product category to validate.

        Returns:
            True if valid, False otherwise.
        """
        return category.lower().strip() in VALID_PRODUCT_CATEGORIES

    def validate_region(self, region: str) -> bool:
        """
        Check if a region code is valid.

        Args:
            region: Region code to validate.

        Returns:
            True if valid, False otherwise.
        """
        return region.upper().strip() in VALID_REGIONS

    def validate_climate_zone(self, climate_zone: str) -> bool:
        """
        Check if a climate zone is valid.

        Args:
            climate_zone: Climate zone to validate.

        Returns:
            True if valid, False otherwise.
        """
        return climate_zone.lower().strip() in VALID_CLIMATE_ZONES

    def validate_landfill_type(self, landfill_type: str) -> bool:
        """
        Check if a landfill type is valid.

        Args:
            landfill_type: Landfill type to validate.

        Returns:
            True if valid, False otherwise.
        """
        return landfill_type.lower().strip() in VALID_LANDFILL_TYPES

    # ==================================================================
    # 16. ENUMERATION ACCESSORS
    # ==================================================================

    def get_valid_materials(self) -> List[str]:
        """Return sorted list of valid material types."""
        return sorted(VALID_MATERIALS)

    def get_valid_treatments(self) -> List[str]:
        """Return sorted list of valid treatment pathways."""
        return sorted(VALID_TREATMENTS)

    def get_valid_product_categories(self) -> List[str]:
        """Return sorted list of valid product categories."""
        return sorted(VALID_PRODUCT_CATEGORIES)

    def get_valid_regions(self) -> List[str]:
        """Return sorted list of valid region codes."""
        return sorted(VALID_REGIONS)

    def get_valid_climate_zones(self) -> List[str]:
        """Return sorted list of valid climate zones."""
        return sorted(VALID_CLIMATE_ZONES)

    def get_valid_landfill_types(self) -> List[str]:
        """Return sorted list of valid landfill types."""
        return sorted(VALID_LANDFILL_TYPES)

    # ==================================================================
    # 17. PROVENANCE / HASHING
    # ==================================================================

    def compute_lookup_hash(
        self,
        material: str,
        treatment: str,
        ef_value: Decimal,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a factor lookup.

        Creates an immutable audit record of which factor was selected,
        when, and from which database version.

        Args:
            material: Material type.
            treatment: Treatment pathway.
            ef_value: The emission factor value returned.

        Returns:
            SHA-256 hex digest string.

        Example:
            >>> h = engine.compute_lookup_hash("plastic", "landfill", Decimal("0.021"))
            >>> len(h)
            64
        """
        payload = json.dumps(
            {
                "agent_id": AGENT_ID,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "material": material,
                "treatment": treatment,
                "ef_value": str(ef_value),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_composite_hash(
        self,
        product_category: str,
        region: str,
        composite_data: Dict[str, Any],
    ) -> str:
        """
        Compute SHA-256 hash for a composite lookup result.

        Args:
            product_category: Product category queried.
            region: Region code queried.
            composite_data: The composite lookup result.

        Returns:
            SHA-256 hex digest string.
        """
        # Convert Decimal values to strings for JSON serialization
        def _decimal_default(obj: Any) -> str:
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        payload = json.dumps(
            {
                "agent_id": AGENT_ID,
                "engine_id": ENGINE_ID,
                "engine_version": ENGINE_VERSION,
                "product_category": product_category,
                "region": region,
                "data": composite_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
            default=_decimal_default,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ==================================================================
    # 18. STATISTICS / METRICS
    # ==================================================================

    def get_lookup_count(self) -> int:
        """
        Get total number of factor lookups performed.

        Returns:
            Count of lookups since engine initialization.
        """
        with self._rlock:
            return self._lookup_count

    def get_last_access(self) -> Optional[datetime]:
        """
        Get timestamp of last access.

        Returns:
            UTC datetime of last access, or None if never accessed.
        """
        with self._rlock:
            return self._last_access

    def get_database_stats(self) -> Dict[str, int]:
        """
        Get counts of records in all embedded databases.

        Returns:
            Dict with counts for each database table.

        Example:
            >>> stats = engine.get_database_stats()
            >>> stats["material_emission_factors"]
            15
            >>> stats["product_compositions"]
            20
        """
        return {
            "material_emission_factors": len(MATERIAL_EMISSION_FACTORS),
            "product_compositions": len(PRODUCT_COMPOSITIONS),
            "regional_treatment_mixes": len(REGIONAL_TREATMENT_MIXES),
            "incineration_params": len(INCINERATION_PARAMS),
            "gas_collection_efficiency": len(GAS_COLLECTION_EFFICIENCY),
            "energy_recovery_factors": len(ENERGY_RECOVERY_FACTORS),
            "recycling_processing_efs": len(RECYCLING_PROCESSING_EFS),
            "avoided_emission_factors": len(AVOIDED_EMISSION_FACTORS),
            "product_weight_defaults": len(PRODUCT_WEIGHT_DEFAULTS),
            "calorific_values": len(CALORIFIC_VALUES),
            "open_burning_efs": len(OPEN_BURNING_EFS),
            "decay_rate_climate_zones": len(DECAY_RATE_CONSTANTS),
            "material_doc_values": len(MATERIAL_DOC_VALUES),
            "mcf_landfill_types": len(MCF_BY_LANDFILL_TYPE),
            "gwp_versions": len(GWP_VALUES),
            "total_lookups_performed": self.get_lookup_count(),
        }

    # ==================================================================
    # 19. HEALTH CHECK
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the database engine.

        Validates that:
        - All databases are loaded and non-empty
        - Singleton is properly initialized
        - Sample lookups succeed
        - Data integrity checks pass

        Returns:
            Dict with status ("healthy" or "unhealthy"), checks performed,
            and any errors encountered.

        Example:
            >>> result = engine.health_check()
            >>> result["status"]
            'healthy'
        """
        start_ts = time.monotonic()
        checks: Dict[str, Any] = {}
        errors: List[str] = []

        # Check 1: Singleton initialized
        checks["singleton_initialized"] = self._initialized
        if not self._initialized:
            errors.append("Engine not initialized")

        # Check 2: All databases non-empty
        stats = self.get_database_stats()
        for db_name, count in stats.items():
            if db_name == "total_lookups_performed":
                continue
            if count == 0:
                errors.append(f"Empty database: {db_name}")
        checks["databases_loaded"] = len(errors) == 0

        # Check 3: Sample lookup - material EF
        try:
            ef = self.get_material_ef("plastic", "landfill")
            checks["sample_material_ef"] = ef == Decimal("0.021")
        except Exception as e:
            checks["sample_material_ef"] = False
            errors.append(f"Material EF lookup failed: {e}")

        # Check 4: Sample lookup - product composition
        try:
            comp = self.get_product_composition("electronics")
            checks["sample_composition"] = len(comp) > 0
        except Exception as e:
            checks["sample_composition"] = False
            errors.append(f"Composition lookup failed: {e}")

        # Check 5: Sample lookup - regional mix
        try:
            mix = self.get_regional_treatment_mix("US")
            checks["sample_regional_mix"] = abs(sum(mix.values()) - _ONE) < Decimal("0.001")
        except Exception as e:
            checks["sample_regional_mix"] = False
            errors.append(f"Regional mix lookup failed: {e}")

        # Check 6: Sample lookup - landfill FOD params
        try:
            fod = self.get_landfill_fod_params("paper", "temperate_wet")
            checks["sample_fod_params"] = fod["doc"] > _ZERO
        except Exception as e:
            checks["sample_fod_params"] = False
            errors.append(f"FOD params lookup failed: {e}")

        # Check 7: Sample lookup - incineration params
        try:
            incin = self.get_incineration_params("plastic")
            checks["sample_incineration"] = incin["fossil_carbon_fraction"] == _ONE
        except Exception as e:
            checks["sample_incineration"] = False
            errors.append(f"Incineration params lookup failed: {e}")

        # Check 8: Data integrity
        try:
            self._validate_data_integrity()
            checks["data_integrity"] = True
        except ValueError as e:
            checks["data_integrity"] = False
            errors.append(f"Data integrity check failed: {e}")

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        status = "healthy" if len(errors) == 0 else "unhealthy"

        result = {
            "status": status,
            "agent_id": AGENT_ID,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "checks": checks,
            "errors": errors,
            "database_stats": stats,
            "elapsed_ms": round(elapsed_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if status == "healthy":
            logger.info(
                "Health check PASSED: engine=%s, checks=%d, elapsed_ms=%.2f",
                ENGINE_ID, len(checks), elapsed_ms,
            )
        else:
            logger.warning(
                "Health check FAILED: engine=%s, errors=%s",
                ENGINE_ID, errors,
            )

        return result

    # ==================================================================
    # 20. STRING REPRESENTATION
    # ==================================================================

    def __repr__(self) -> str:
        """Return string representation of the engine."""
        return (
            f"EOLProductDatabaseEngine("
            f"agent_id={AGENT_ID!r}, "
            f"engine_version={ENGINE_VERSION!r}, "
            f"materials={len(MATERIAL_EMISSION_FACTORS)}, "
            f"products={len(PRODUCT_COMPOSITIONS)}, "
            f"regions={len(REGIONAL_TREATMENT_MIXES)}, "
            f"lookups={self.get_lookup_count()}"
            f")"
        )

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"EOLProductDatabaseEngine v{ENGINE_VERSION} "
            f"({AGENT_ID}): "
            f"{len(MATERIAL_EMISSION_FACTORS)} materials, "
            f"{len(PRODUCT_COMPOSITIONS)} products, "
            f"{len(REGIONAL_TREATMENT_MIXES)} regions"
        )
