# -*- coding: utf-8 -*-
"""
End-of-Life Treatment of Sold Products Agent Package - AGENT-MRV-025

GHG Protocol Scope 3, Category 12: End-of-Life Treatment of Sold Products.
Calculates emissions from the waste disposal and treatment of products sold
by the reporting company (in the reporting year) at the end of their life.

Agent ID: GL-MRV-S3-012
Package: greenlang.agents.mrv.end_of_life_treatment
API: /api/v1/end-of-life-treatment
DB Migration: V076
Metrics Prefix: gl_eol_
Table Prefix: gl_eol_

Supported Treatment Pathways:
    - Landfill (IPCC FOD model with gas collection and oxidation)
    - Incineration (mass burn, RDF, waste-to-energy, open burning)
    - Recycling (cut-off, closed-loop, substitution with avoided credits)
    - Composting (industrial/home, CH4 and N2O process emissions)
    - Anaerobic digestion (biogas capture, fugitive CH4)
    - Open burning (uncontrolled, high emission factors)
    - Wastewater (BOD/COD-based CH4 and effluent N2O)

Calculation Methods:
    - Waste-type-specific (material x treatment x EF per kg)
    - Average-data (product category average EFs)
    - Producer-specific (EPD-based lifecycle end-of-life data)
    - Hybrid (blended multi-method aggregation)

Key Differentiator from Category 5 (Waste Generated):
    Category 5 covers waste from the reporting company's own operations.
    Category 12 covers end-of-life treatment of products SOLD by the company,
    disposed of by downstream consumers and third parties.

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "EOLProductDatabaseEngine",
    "WasteTypeSpecificCalculatorEngine",
    "AverageDataCalculatorEngine",
    "ProducerSpecificCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "EndOfLifeTreatmentPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "PIPELINE_STAGES",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-012"
AGENT_COMPONENT: str = "AGENT-MRV-025"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_eol_"

PIPELINE_STAGES: list = [
    "validate",
    "classify",
    "normalize",
    "resolve_efs",
    "calculate",
    "allocate",
    "aggregate",
    "compliance",
    "provenance",
    "seal",
]

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.end_of_life_treatment.eol_product_database import EOLProductDatabaseEngine
except ImportError:
    EOLProductDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.waste_type_specific_calculator import WasteTypeSpecificCalculatorEngine
except ImportError:
    WasteTypeSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.average_data_calculator import AverageDataCalculatorEngine
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.producer_specific_calculator import ProducerSpecificCalculatorEngine
except ImportError:
    ProducerSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.hybrid_aggregator import HybridAggregatorEngine
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.end_of_life_treatment.end_of_life_treatment_pipeline import EndOfLifeTreatmentPipelineEngine
except ImportError:
    EndOfLifeTreatmentPipelineEngine = None  # type: ignore[assignment,misc]

# Export agent metadata from models
try:
    from greenlang.agents.mrv.end_of_life_treatment.models import (
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION as MODELS_VERSION,
        TABLE_PREFIX,
    )
except ImportError:
    AGENT_ID = "GL-MRV-S3-012"
    AGENT_COMPONENT = "AGENT-MRV-025"
    TABLE_PREFIX = "gl_eol_"

# Export configuration helper
try:
    from greenlang.agents.mrv.end_of_life_treatment.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
