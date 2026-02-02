"""
GL-016 WATERGUARD - Industrial Water Treatment Agent
Integrations Module

Provides integration capabilities for water treatment system monitoring,
control, and optimization including:
- SCADA/DCS integration for water analyzers and chemical dosing
- Water analyzer integration (Hach, Mettler Toledo, Emerson, E+H)
- Chemical dosing system control
- ERP integration for inventory and compliance
- Multi-agent coordination with GL-001 through GL-015
- Data transformation and validation

Author: GreenLang Integrations Engineering Team
Version: 1.0.0
License: Proprietary
"""

from .scada_integration import (
    SCADAClient,
    SCADAConfig,
    WaterQualityTag,
    TagDataPoint,
    create_scada_client,
)

from .water_analyzer_integration import (
    WaterAnalyzerClient,
    WaterAnalyzerConfig,
    AnalyzerBrand,
    AnalyzerType,
    AnalyzerReading,
    CalibrationStatus,
    create_water_analyzer_client,
)

from .chemical_dosing_integration import (
    ChemicalDosingController,
    ChemicalDosingConfig,
    ChemicalType,
    DosingPumpType,
    DosingCommand,
    ChemicalInventoryStatus,
    create_dosing_controller,
)

from .erp_connector import (
    ERPClient,
    ERPConfig,
    ERPSystem,
    PurchaseOrder,
    ChemicalInventory,
    ComplianceReport,
    create_erp_client,
)

from .agent_coordinator import (
    AgentCoordinator,
    AgentCoordinatorConfig,
    AgentID,
    MessageType,
    AgentMessage,
    AgentResponse,
    create_agent_coordinator,
)

from .data_transformers import (
    DataTransformer,
    UnitConverter,
    TimestampNormalizer,
    SchemaMapper,
    DataQualityScorer,
    MissingDataHandler,
    OutlierDetector,
)

__version__ = "1.0.0"
__author__ = "GreenLang Integrations Engineering Team"

__all__ = [
    # SCADA Integration
    "SCADAClient",
    "SCADAConfig",
    "WaterQualityTag",
    "TagDataPoint",
    "create_scada_client",

    # Water Analyzer Integration
    "WaterAnalyzerClient",
    "WaterAnalyzerConfig",
    "AnalyzerBrand",
    "AnalyzerType",
    "AnalyzerReading",
    "CalibrationStatus",
    "create_water_analyzer_client",

    # Chemical Dosing Integration
    "ChemicalDosingController",
    "ChemicalDosingConfig",
    "ChemicalType",
    "DosingPumpType",
    "DosingCommand",
    "ChemicalInventoryStatus",
    "create_dosing_controller",

    # ERP Integration
    "ERPClient",
    "ERPConfig",
    "ERPSystem",
    "PurchaseOrder",
    "ChemicalInventory",
    "ComplianceReport",
    "create_erp_client",

    # Agent Coordination
    "AgentCoordinator",
    "AgentCoordinatorConfig",
    "AgentID",
    "MessageType",
    "AgentMessage",
    "AgentResponse",
    "create_agent_coordinator",

    # Data Transformers
    "DataTransformer",
    "UnitConverter",
    "TimestampNormalizer",
    "SchemaMapper",
    "DataQualityScorer",
    "MissingDataHandler",
    "OutlierDetector",
]


# Module metadata
MODULE_INFO = {
    "name": "GL-016 WATERGUARD Integrations",
    "version": __version__,
    "description": "Industrial water treatment system integration module",
    "author": __author__,
    "capabilities": [
        "SCADA/DCS integration for water quality monitoring",
        "Multi-brand water analyzer support",
        "Chemical dosing system control",
        "ERP integration for inventory and compliance",
        "Multi-agent coordination",
        "Data transformation and quality validation",
    ],
    "supported_systems": {
        "scada": ["OPC-UA", "Modbus TCP", "Profinet"],
        "analyzers": ["Hach", "Mettler Toledo", "Emerson", "Endress+Hauser"],
        "erp": ["SAP", "Oracle", "Microsoft Dynamics"],
        "protocols": ["OPC-UA", "Modbus RTU/TCP", "HART", "Profibus"],
    },
}


def get_module_info() -> dict:
    """
    Get integration module information.

    Returns:
        Dictionary containing module metadata
    """
    return MODULE_INFO.copy()


def validate_integration_environment() -> dict:
    """
    Validate that all required integration dependencies are available.

    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "missing_packages": [],
        "warnings": [],
    }

    # Check for optional packages
    optional_packages = {
        "opcua": "asyncua",
        "modbus": "pymodbus",
        "redis": "redis",
        "sqlalchemy": "sqlalchemy",
    }

    for name, package in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            results["missing_packages"].append(package)
            results["warnings"].append(
                f"{package} not installed - {name} integration will not work"
            )

    if results["missing_packages"]:
        results["valid"] = False

    return results
