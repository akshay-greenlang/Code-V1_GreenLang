# -*- coding: utf-8 -*-
"""
Processing of Sold Products Agent Package - AGENT-MRV-023

GHG Protocol Scope 3, Category 10: Processing of Sold Products.
Calculates emissions from downstream processing of intermediate products
sold by the reporting company, where such processing is not controlled
by the reporting company.

Agent ID: GL-MRV-S3-010
Package: greenlang.agents.mrv.processing_sold_products
API: /api/v1/processing-sold-products
DB Migration: V074
Metrics Prefix: gl_psp_
Table Prefix: gl_psp_

Supported Intermediate Product Categories:
    - Metals (ferrous and non-ferrous)
    - Plastics (thermoplastic and thermoset)
    - Chemicals
    - Food ingredients
    - Textiles
    - Electronics components
    - Glass and ceramics
    - Wood and paper pulp
    - Minerals
    - Agricultural commodities

Processing Types (18):
    - Machining, stamping, welding, heat treatment
    - Injection molding, extrusion, blow molding
    - Casting, forging, coating, assembly
    - Chemical reaction, refining, milling
    - Drying, sintering, fermentation, textile finishing

Calculation Methods:
    - Site-specific direct (customer-reported processing emissions)
    - Site-specific energy (energy consumption x grid/fuel EF)
    - Site-specific fuel (fuel consumption x combustion EF)
    - Average-data (product category x processing type EF)
    - Spend-based (revenue x EEIO sector factor)

Frameworks:
    - GHG Protocol Scope 3 Category 10
    - ISO 14064-1:2018
    - CSRD ESRS E1 Climate Change
    - CDP Climate Change Questionnaire
    - SBTi Corporate Net-Zero
    - SB 253 (California Climate Disclosure)
    - GRI 305 Emissions Standard

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "ProcessingDatabaseEngine",
    "SiteSpecificCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "ProcessingPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "PIPELINE_STAGES",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-010"
AGENT_COMPONENT: str = "AGENT-MRV-023"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_psp_"

PIPELINE_STAGES: list = [
    "VALIDATE",
    "CLASSIFY",
    "NORMALIZE",
    "RESOLVE_EFS",
    "CALCULATE",
    "ALLOCATE",
    "AGGREGATE",
    "COMPLIANCE",
    "PROVENANCE",
    "SEAL",
]

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.processing_sold_products.processing_database import (
        ProcessingDatabaseEngine,
    )
except ImportError:
    ProcessingDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.site_specific_calculator import (
        SiteSpecificCalculatorEngine,
    )
except ImportError:
    SiteSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.processing_sold_products.processing_pipeline import (
        ProcessingPipelineEngine,
    )
except ImportError:
    ProcessingPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.agents.mrv.processing_sold_products.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
