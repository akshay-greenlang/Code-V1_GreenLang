# -*- coding: utf-8 -*-
"""
Waste Generated in Operations Agent Package - AGENT-MRV-018

GHG Protocol Scope 3, Category 5: Waste Generated in Operations.
Calculates emissions from treatment and disposal of waste generated
in the reporting company's facilities and offices.

Agent ID: GL-MRV-S3-005
Package: greenlang.agents.mrv.waste_generated
API: /api/v1/waste-generated
DB Migration: V069
Metrics Prefix: gl_wg_
Table Prefix: gl_wg_

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "WasteClassificationDatabaseEngine",
    "LandfillEmissionsEngine",
    "IncinerationEmissionsEngine",
    "RecyclingCompostingEngine",
    "WastewaterEmissionsEngine",
    "ComplianceCheckerEngine",
    "WasteGeneratedPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "get_config",
]

VERSION: str = "1.0.0"

# Graceful imports - each engine with try/except
try:
    from greenlang.agents.mrv.waste_generated.waste_classification_database import WasteClassificationDatabaseEngine
except ImportError:
    WasteClassificationDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.landfill_emissions import LandfillEmissionsEngine
except ImportError:
    LandfillEmissionsEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.incineration_emissions import IncinerationEmissionsEngine
except ImportError:
    IncinerationEmissionsEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.recycling_composting import RecyclingCompostingEngine
except ImportError:
    RecyclingCompostingEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.wastewater_emissions import WastewaterEmissionsEngine
except ImportError:
    WastewaterEmissionsEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.mrv.waste_generated.waste_generated_pipeline import WasteGeneratedPipelineEngine
except ImportError:
    WasteGeneratedPipelineEngine = None  # type: ignore[assignment,misc]

# Export agent metadata from models
try:
    from greenlang.agents.mrv.waste_generated.models import AGENT_ID, AGENT_COMPONENT, VERSION as MODELS_VERSION, TABLE_PREFIX
except ImportError:
    AGENT_ID = "GL-MRV-S3-005"
    AGENT_COMPONENT = "AGENT-MRV-018"
    TABLE_PREFIX = "gl_wg_"

# Export configuration helper
try:
    from greenlang.agents.mrv.waste_generated.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
