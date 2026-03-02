# -*- coding: utf-8 -*-
"""
Use of Sold Products Agent Package - AGENT-MRV-024

GHG Protocol Scope 3, Category 11: Use of Sold Products.
Calculates total expected lifetime emissions from the USE of goods and
services sold by the reporting company in the reporting period. This is
often the largest Scope 3 category for manufacturers of energy-consuming
products (vehicles, appliances, electronics, HVAC).

Agent ID: GL-MRV-S3-011
Package: greenlang.use_of_sold_products
API: /api/v1/use-of-sold-products
DB Migration: V075
Metrics Prefix: gl_usp_
Table Prefix: gl_usp_

Two Emission Types:
    - Direct use-phase emissions: GHG released directly during product use
      (fuel combustion in vehicles, refrigerant leakage from HVAC,
      chemical release from consumer products)
    - Indirect use-phase emissions: GHG from generation of energy consumed
      during product use (electricity for appliances, heating fuel for
      furnaces, district steam/cooling)

Product Use Categories (10):
    - VEHICLES: Cars, trucks, motorcycles (direct fuel combustion)
    - APPLIANCES: Refrigerators, washing machines, ovens (indirect electricity)
    - HVAC: Air conditioners, heat pumps, furnaces (direct + indirect)
    - LIGHTING: LED bulbs, CFL bulbs (indirect electricity)
    - IT_EQUIPMENT: Laptops, desktops, servers, monitors (indirect electricity)
    - INDUSTRIAL_EQUIPMENT: Generators, boilers, compressors (direct + indirect)
    - FUELS_FEEDSTOCKS: Gasoline, diesel, natural gas, coal (direct combustion)
    - BUILDING_PRODUCTS: Windows, insulation, HVAC ducts (indirect)
    - CONSUMER_PRODUCTS: Aerosols, solvents, fertilizers (direct chemical)
    - MEDICAL_DEVICES: Imaging equipment, ventilators (indirect electricity)

Calculation Methods (8):
    - Direct fuel combustion (vehicles, generators)
    - Direct refrigerant leakage (HVAC, refrigeration)
    - Direct chemical release (consumer chemicals)
    - Indirect electricity consumption (appliances, IT, lighting)
    - Indirect heating fuel (furnaces, boilers)
    - Indirect steam/cooling (district energy)
    - Fuels sold (fuel combustion by end users)
    - Feedstocks sold (feedstock oxidation)

Frameworks:
    - GHG Protocol Scope 3 Category 11 (Chapter 6)
    - ISO 14064-1:2018
    - CSRD ESRS E1
    - CDP Climate Change
    - SBTi Corporate Net-Zero
    - SB 253 (California Climate Disclosure)
    - GRI 305

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "ProductUseDatabaseEngine",
    "DirectEmissionsCalculatorEngine",
    "IndirectEmissionsCalculatorEngine",
    "FuelsAndFeedstocksCalculatorEngine",
    "LifetimeModelingEngine",
    "ComplianceCheckerEngine",
    "UseOfSoldProductsPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "PIPELINE_STAGES",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_usp_"

PIPELINE_STAGES: list = [
    "validate",
    "classify",
    "normalize",
    "resolve_efs",
    "calculate",
    "lifetime",
    "aggregate",
    "compliance",
    "provenance",
    "seal",
]

# Graceful imports - each engine with try/except
try:
    from greenlang.use_of_sold_products.product_use_database import (
        ProductUseDatabaseEngine,
    )
except ImportError:
    ProductUseDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.direct_emissions_calculator import (
        DirectEmissionsCalculatorEngine,
    )
except ImportError:
    DirectEmissionsCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.indirect_emissions_calculator import (
        IndirectEmissionsCalculatorEngine,
    )
except ImportError:
    IndirectEmissionsCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.fuels_feedstocks_calculator import (
        FuelsAndFeedstocksCalculatorEngine,
    )
except ImportError:
    FuelsAndFeedstocksCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.lifetime_modeling import (
        LifetimeModelingEngine,
    )
except ImportError:
    LifetimeModelingEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
        UseOfSoldProductsPipelineEngine,
    )
except ImportError:
    UseOfSoldProductsPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.use_of_sold_products.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
